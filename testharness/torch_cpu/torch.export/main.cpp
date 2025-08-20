#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
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
        
        // Try different export options based on remaining data
        if (offset < Size) {
            uint8_t export_option = Data[offset++] % 3;
            
            try {
                if (export_option == 0) {
                    // Basic export
                    torch::jit::Module exported_module = torch::jit::tracer::trace(model, example_inputs);
                } else if (export_option == 1) {
                    // Export with optimization
                    torch::jit::Module exported_module = torch::jit::tracer::trace(model, example_inputs);
                    exported_module = torch::jit::optimize_for_inference(exported_module);
                } else {
                    // Export with custom attributes
                    torch::jit::Module exported_module = torch::jit::tracer::trace(model, example_inputs);
                    
                    // Try to add custom attributes if we have more data
                    if (offset < Size) {
                        uint8_t attr_type = Data[offset++] % 4;
                        
                        if (attr_type == 0) {
                            exported_module.setattr("int_attr", torch::tensor(static_cast<int64_t>(Data[offset % Size])));
                        } else if (attr_type == 1) {
                            exported_module.setattr("float_attr", torch::tensor(static_cast<float>(Data[offset % Size]) / 255.0f));
                        } else if (attr_type == 2) {
                            exported_module.setattr("bool_attr", torch::tensor(Data[offset % Size] > 127));
                        } else {
                            // Create a small tensor attribute
                            torch::Tensor attr_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                            exported_module.setattr("tensor_attr", attr_tensor);
                        }
                    }
                }
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors
                return 0;
            }
        } else {
            // Basic export if no more data
            torch::jit::Module exported_module = torch::jit::tracer::trace(model, example_inputs);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}