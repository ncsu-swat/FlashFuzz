#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>

class SimpleModel : public torch::nn::Module {
public:
    SimpleModel() {
        linear = register_module("linear", torch::nn::Linear(10, 5));
    }

    torch::Tensor forward(torch::Tensor x) {
        return linear(x);
    }

private:
    torch::nn::Linear linear{nullptr};
};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple model
        auto model = std::make_shared<SimpleModel>();
        
        // Create an example input for tracing
        std::vector<torch::jit::IValue> example_inputs;
        example_inputs.push_back(input_tensor);
        
        // Trace the model
        try {
            torch::jit::Module traced_module = torch::jit::trace(model, example_inputs);
            
            // Test the traced module with the same input
            std::vector<torch::jit::IValue> test_inputs;
            test_inputs.push_back(input_tensor);
            
            // Run the traced module
            torch::Tensor output = traced_module.forward(test_inputs).toTensor();
            
            // Try to save and load the traced module
            if (offset < Size) {
                uint8_t save_flag = Data[offset++];
                if (save_flag % 2 == 0) {
                    traced_module.save("temp_module.pt");
                    
                    // Load the saved module
                    torch::jit::Module loaded_module = torch::jit::load("temp_module.pt");
                    
                    // Run the loaded module
                    torch::Tensor loaded_output = loaded_module.forward(test_inputs).toTensor();
                }
            }
            
            // Try to run the module in different modes
            if (offset < Size) {
                uint8_t mode_flag = Data[offset++];
                if (mode_flag % 3 == 0) {
                    traced_module.eval();
                } else if (mode_flag % 3 == 1) {
                    traced_module.train();
                }
                
                // Run again after changing mode
                torch::Tensor mode_output = traced_module.forward(test_inputs).toTensor();
            }
            
            // Try to get attributes from the module
            if (offset < Size) {
                uint8_t attr_flag = Data[offset++];
                if (attr_flag % 2 == 0) {
                    try {
                        auto attr_list = traced_module.attributes();
                        for (const auto& attr : attr_list) {
                            auto attr_value = traced_module.attr(attr.name);
                        }
                    } catch (...) {
                        // Ignore attribute access errors
                    }
                }
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors during tracing
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}