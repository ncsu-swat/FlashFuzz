#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>

// Simple model class for tracing
struct TestModel : torch::nn::Module {
    TestModel() {}
    
    torch::Tensor forward(torch::Tensor x) {
        return x * 2;
    }
};

// Function to create a simple model
torch::jit::Module createSimpleModel(torch::Tensor example_input) {
    TestModel model;
    try {
        return torch::jit::trace(model, example_input);
    } catch (const c10::Error& e) {
        throw;
    }
}

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to work with
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for tracing
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to trace a model with this input
        try {
            torch::jit::Module traced_module = createSimpleModel(input_tensor);
            
            // Test the traced model with the same input
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            torch::Tensor output = traced_module.forward(inputs).toTensor();
            
            // Try to save and load the traced model
            if (offset + 1 < Size) {
                uint8_t save_flag = Data[offset++];
                if (save_flag % 2 == 0) {
                    traced_module.save("temp_model.pt");
                    torch::jit::Module loaded_module = torch::jit::load("temp_model.pt");
                    std::vector<torch::jit::IValue> loaded_inputs;
                    loaded_inputs.push_back(input_tensor);
                    torch::Tensor loaded_output = loaded_module.forward(loaded_inputs).toTensor();
                }
            }
            
            // Try with a different input if we have enough data
            if (offset < Size) {
                try {
                    torch::Tensor second_input = fuzzer_utils::createTensor(Data, Size, offset);
                    if (second_input.sizes() == input_tensor.sizes()) {
                        std::vector<torch::jit::IValue> second_inputs;
                        second_inputs.push_back(second_input);
                        torch::Tensor second_output = traced_module.forward(second_inputs).toTensor();
                    }
                } catch (...) {
                    // Ignore errors with the second input
                }
            }
            
        } catch (const c10::Error& e) {
            // Expected PyTorch errors during tracing are acceptable
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
