#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>
#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/tracer.h>

class SimpleModel : public torch::nn::Module {
public:
    SimpleModel(int64_t in_features, int64_t out_features) 
        : linear(torch::nn::LinearOptions(in_features, out_features)) {
        register_module("linear", linear);
    }

    torch::Tensor forward(torch::Tensor x) {
        return linear(x);
    }

private:
    torch::nn::Linear linear{nullptr};
};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract model configuration from fuzzer data
        int64_t in_features = static_cast<int64_t>((Data[offset++] % 16) + 1);  // 1-16
        int64_t out_features = static_cast<int64_t>((Data[offset++] % 16) + 1); // 1-16
        uint8_t batch_size_raw = Data[offset++];
        int64_t batch_size = static_cast<int64_t>((batch_size_raw % 8) + 1);    // 1-8
        uint8_t export_option = Data[offset++] % 4;
        uint8_t attr_type = Data[offset++] % 5;
        
        // Create model with fuzz-controlled dimensions
        SimpleModel model(in_features, out_features);
        model.eval();
        
        // Create input tensor with compatible shape [batch_size, in_features]
        torch::Tensor input_tensor = torch::randn({batch_size, in_features});
        
        // If we have enough data, use it to influence tensor values
        if (offset + 4 <= Size) {
            float scale = static_cast<float>(Data[offset++]) / 255.0f * 10.0f;
            float shift = static_cast<float>(Data[offset++]) / 255.0f * 5.0f - 2.5f;
            input_tensor = input_tensor * scale + shift;
        }
        
        std::vector<torch::jit::IValue> example_inputs;
        example_inputs.push_back(input_tensor);

        auto trace_once = [&](bool force_outplace, bool strict) {
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
                /*strict=*/strict,
                /*force_outplace=*/force_outplace,
                /*self=*/nullptr,
                {});

            torch::jit::Stack traced_outputs = std::get<1>(traced);
            if (!traced_outputs.empty() && traced_outputs[0].isTensor()) {
                torch::Tensor out = traced_outputs[0].toTensor();
                // Exercise the output
                out.sum();
                out.mean();
            }
            return std::get<0>(traced); // Return the traced graph
        };
        
        try {
            if (export_option == 0) {
                // Basic tracing with strict mode
                trace_once(false, true);
            } else if (export_option == 1) {
                // Tracing with force_outplace
                trace_once(true, true);
            } else if (export_option == 2) {
                // Non-strict tracing
                trace_once(false, false);
            } else {
                // Trace with different configurations and add attributes
                trace_once(false, true);
                
                if (attr_type == 0) {
                    int64_t val = static_cast<int64_t>(Data[offset % Size]);
                    model.register_buffer("int_attr", torch::tensor(val));
                } else if (attr_type == 1) {
                    float val = static_cast<float>(Data[offset % Size]) / 255.0f;
                    model.register_buffer("float_attr", torch::tensor(val));
                } else if (attr_type == 2) {
                    bool val = Data[offset % Size] > 127;
                    model.register_buffer("bool_attr", torch::tensor(val));
                } else if (attr_type == 3) {
                    // Create a small tensor buffer
                    torch::Tensor attr_tensor = torch::randn({4});
                    model.register_buffer("tensor_attr", attr_tensor);
                } else {
                    // Multiple buffers
                    model.register_buffer("buf1", torch::zeros({2, 2}));
                    model.register_buffer("buf2", torch::ones({3}));
                }
                
                // Trace again after adding attributes
                trace_once(true, false);
            }
        } catch (const c10::Error& e) {
            // Expected PyTorch errors during tracing - silently ignore
        } catch (const std::runtime_error& e) {
            // Expected runtime errors during tracing - silently ignore
        }
        
        // Exercise additional tracing paths if we have more data
        if (offset < Size && Data[offset] % 2 == 0) {
            try {
                // Try tracing with a different input shape
                int64_t new_batch = static_cast<int64_t>((Data[offset] % 4) + 1);
                torch::Tensor new_input = torch::randn({new_batch, in_features});
                example_inputs.clear();
                example_inputs.push_back(new_input);
                trace_once(false, true);
            } catch (const c10::Error& e) {
                // Silently ignore shape-related errors
            } catch (const std::runtime_error& e) {
                // Silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}