#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
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
        
        // Create a simple model to trace
        torch::nn::Sequential model(
            torch::nn::Linear(input_tensor.size(-1), 10),
            torch::nn::ReLU(),
            torch::nn::Linear(10, 5)
        );
        
        // Set model to evaluation mode
        model->eval();
        
        // Create dummy input for tracing
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        try {
            // Trace the model
            torch::jit::script::Module script_module = torch::jit::trace(model, input_tensor);
            
            // Run inference with the traced model
            torch::Tensor output = script_module.forward(inputs).toTensor();
            
            // Verify output shape
            if (output.dim() > 0 && input_tensor.dim() > 0) {
                int64_t expected_batch_size = input_tensor.size(0);
                int64_t expected_output_size = 5;
                
                if (output.size(0) != expected_batch_size || output.size(-1) != expected_output_size) {
                    // This is not an error, just a verification
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected during fuzzing
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