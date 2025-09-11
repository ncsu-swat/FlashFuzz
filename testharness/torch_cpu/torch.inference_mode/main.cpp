#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to use with inference_mode
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a boolean value from the data to determine if we should enable or disable inference mode
        bool enable_inference_mode = offset < Size ? (Data[offset++] % 2 == 0) : true;
        
        // Test inference_mode with the tensor
        {
            // Enter inference mode scope
            torch::InferenceMode guard(enable_inference_mode);
            
            // Perform some operations on the tensor inside inference mode
            torch::Tensor result1 = tensor + 1;
            torch::Tensor result2 = tensor * 2;
            torch::Tensor result3 = torch::relu(tensor);
            
            // Check if inference mode is active
            bool is_inference_mode_enabled = torch::InferenceMode::is_enabled();
            
            // Verify that inference mode status matches what we set
            if (is_inference_mode_enabled != enable_inference_mode) {
                throw std::runtime_error("InferenceMode status mismatch");
            }
            
            // Perform more operations to test inference mode behavior
            torch::Tensor result4 = torch::softmax(tensor, 0);
            torch::Tensor result5 = torch::log_softmax(tensor, 0);
            
            // Test nested inference mode scopes
            {
                // Create a nested inference mode scope with opposite setting
                torch::InferenceMode nested_guard(!enable_inference_mode);
                
                // Check if the nested scope overrides the outer scope
                bool nested_inference_mode_enabled = torch::InferenceMode::is_enabled();
                
                // Perform operations in nested scope
                torch::Tensor nested_result = tensor.clone();
                
                // Exit nested scope
            }
            
            // Verify that we're back to the original inference mode setting
            bool after_nested_inference_mode_enabled = torch::InferenceMode::is_enabled();
            if (after_nested_inference_mode_enabled != enable_inference_mode) {
                throw std::runtime_error("InferenceMode status not restored after nested scope");
            }
            
            // Exit inference mode scope
        }
        
        // Verify that inference mode is disabled after exiting all scopes
        bool final_inference_mode_enabled = torch::InferenceMode::is_enabled();
        if (final_inference_mode_enabled) {
            throw std::runtime_error("InferenceMode still enabled after exiting all scopes");
        }
        
        // Test with requires_grad
        if (offset < Size) {
            bool requires_grad = Data[offset++] % 2 == 0;
            
            // Create a tensor with requires_grad
            torch::Tensor grad_tensor = tensor.clone().detach().requires_grad_(requires_grad);
            
            // Test inference mode with requires_grad tensor
            {
                torch::InferenceMode inference_guard(enable_inference_mode);
                
                // Perform operations
                torch::Tensor grad_result = grad_tensor + 1;
                
                // Check if grad_result has requires_grad
                bool has_grad = grad_result.requires_grad();
                
                // In inference mode, requires_grad should be false regardless of input
                if (enable_inference_mode && has_grad) {
                    throw std::runtime_error("Tensor has requires_grad=true in inference mode");
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
