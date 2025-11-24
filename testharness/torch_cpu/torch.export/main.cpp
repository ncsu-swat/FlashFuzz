#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/tracer.h>

class SimpleModel : public torch::nn::Module {
public:
    SimpleModel() {
        register_module("linear", linear);
    }

    torch::Tensor forward(torch::Tensor x) {
        return linear(x);
    }

private:
    torch::nn::Linear linear{torch::nn::LinearOptions(10, 5)};
};

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // torch.export harness: exercise export-like tracing paths.
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple model
        SimpleModel model;
        model.eval();
        
        // Create a sample input for tracing
        std::vector<torch::jit::IValue> example_inputs;
        example_inputs.push_back(input_tensor);

        auto trace_once = [&](bool force_outplace) {
            torch::jit::Stack input_stack(example_inputs.begin(), example_inputs.end());
            auto traced = torch::jit::tracer::trace(
                std::move(input_stack),
                [&model](torch::jit::Stack inputs) -> torch::jit::Stack {
                    torch::jit::Stack outputs;
                    if (!inputs.empty() && inputs[0].isTensor()) {
                        torch::Tensor x = inputs[0].toTensor();
                        outputs.push_back(model.forward(x));
                    }
                    return outputs;
                },
                [](const torch::autograd::Variable &) { return std::string(); },
                /*strict=*/true,
                /*force_outplace=*/force_outplace,
                /*self=*/nullptr,
                {});

            torch::jit::Stack traced_outputs = std::get<1>(traced);
            if (!traced_outputs.empty() && traced_outputs[0].isTensor()) {
                traced_outputs[0].toTensor().sum();
            }
        };
        
        // Try different export options based on remaining data
        if (offset < Size) {
            uint8_t export_option = Data[offset++] % 3;
            
            try {
                if (export_option == 0) {
                    // Basic export
                    trace_once(false);
                } else if (export_option == 1) {
                    // Export with optimization
                    trace_once(true);
                } else {
                    trace_once(false);

                    // Try to tweak module state if we have more data
                    if (offset < Size) {
                        uint8_t attr_type = Data[offset++] % 4;
                        
                        if (attr_type == 0) {
                            model.register_buffer("int_attr", torch::tensor(static_cast<int64_t>(Data[offset % Size])));
                        } else if (attr_type == 1) {
                            model.register_buffer("float_attr", torch::tensor(static_cast<float>(Data[offset % Size]) / 255.0f));
                        } else if (attr_type == 2) {
                            model.register_buffer("bool_attr", torch::tensor(Data[offset % Size] > 127));
                        } else {
                            // Create a small tensor attribute
                            torch::Tensor attr_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                            model.register_buffer("tensor_attr", attr_tensor);
                        }
                    }
                }
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors
                return 0;
            }
        } else {
            // Basic export if no more data
            trace_once(false);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
