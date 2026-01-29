#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
            
            // Use try-catch for operations that may fail on certain tensor types
            try {
                torch::Tensor result3 = torch::relu(tensor);
            } catch (...) {
                // relu may fail on certain dtypes, silently ignore
            }
            
            // Check if inference mode is active
            bool is_inference_mode_enabled = torch::InferenceMode::is_enabled();
            
            // Perform more operations to test inference mode behavior
            try {
                // softmax requires floating point tensors
                if (tensor.is_floating_point()) {
                    torch::Tensor result4 = torch::softmax(tensor, 0);
                    torch::Tensor result5 = torch::log_softmax(tensor, 0);
                }
            } catch (...) {
                // Silently ignore softmax failures
            }
            
            // Test nested inference mode scopes
            {
                // Create a nested inference mode scope with opposite setting
                torch::InferenceMode nested_guard(!enable_inference_mode);
                
                // Check the nested scope status
                bool nested_inference_mode_enabled = torch::InferenceMode::is_enabled();
                
                // Perform operations in nested scope
                torch::Tensor nested_result = tensor.clone();
                
                // Additional operations in nested scope
                try {
                    torch::Tensor nested_add = nested_result + tensor;
                    torch::Tensor nested_mul = nested_result * 2.0f;
                } catch (...) {
                    // Silently ignore operation failures
                }
                
                // Exit nested scope
            }
            
            // After exiting nested scope, check that outer scope state is restored
            bool after_nested = torch::InferenceMode::is_enabled();
            
            // Exit inference mode scope
        }
        
        // After exiting all inference mode scopes, check the final state
        bool final_inference_mode_enabled = torch::InferenceMode::is_enabled();
        
        // Test with requires_grad
        if (offset < Size) {
            bool requires_grad = Data[offset++] % 2 == 0;
            
            // Only floating point tensors can have requires_grad
            try {
                torch::Tensor grad_tensor;
                if (tensor.is_floating_point()) {
                    grad_tensor = tensor.clone().detach().requires_grad_(requires_grad);
                } else {
                    // Convert to float for requires_grad testing
                    grad_tensor = tensor.to(torch::kFloat32).detach().requires_grad_(requires_grad);
                }
                
                // Test inference mode with requires_grad tensor
                {
                    torch::InferenceMode inference_guard(enable_inference_mode);
                    
                    // Perform operations
                    torch::Tensor grad_result = grad_tensor + 1;
                    
                    // In inference mode, operations don't track gradients
                    // but the tensor's requires_grad property may still be set
                    bool has_grad = grad_result.requires_grad();
                    
                    // Perform more operations
                    torch::Tensor grad_result2 = grad_result * 2;
                    torch::Tensor grad_result3 = torch::abs(grad_result);
                }
            } catch (...) {
                // Silently ignore requires_grad related failures
            }
        }
        
        // Test toggling inference mode multiple times
        if (offset + 2 < Size) {
            int num_toggles = (Data[offset++] % 4) + 1;  // 1-4 toggles
            
            for (int i = 0; i < num_toggles; i++) {
                bool mode = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
                torch::InferenceMode toggle_guard(mode);
                
                // Perform a simple operation in each toggle
                try {
                    torch::Tensor toggled_result = tensor + static_cast<float>(i);
                } catch (...) {
                    // Silently ignore failures
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;  // keep the input
}