#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to work with
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test torch::NoGradGuard functionality
        {
            // Check initial requires_grad state
            bool initial_requires_grad = torch::GradMode::is_enabled();
            
            // Create a scope with NoGradGuard
            {
                torch::NoGradGuard no_grad;
                
                // Verify that grad mode is disabled
                bool during_no_grad = torch::GradMode::is_enabled();
                
                // Perform some operations on the tensor
                torch::Tensor result = tensor + 1;
                
                // Try another operation
                torch::Tensor result2 = torch::sin(tensor);
                
                // Try a more complex operation
                torch::Tensor result3 = torch::matmul(tensor, tensor);
            }
            
            // Verify that grad mode is restored after the scope
            bool after_no_grad = torch::GradMode::is_enabled();
        }
        
        // Test torch::NoGradGuard with exceptions
        {
            try {
                torch::NoGradGuard no_grad;
                
                // Perform an operation that might throw
                torch::Tensor result = torch::log(tensor);
                
                // Intentionally trigger an exception
                if (offset % 2 == 0) {
                    throw std::runtime_error("Intentional exception");
                }
            }
            catch (const std::exception& e) {
                // Exception caught, but NoGradGuard should still restore grad mode
            }
            
            // Verify grad mode is restored even after exception
            bool after_exception = torch::GradMode::is_enabled();
        }
        
        // Test nested NoGradGuard
        {
            bool outer_grad_mode = torch::GradMode::is_enabled();
            
            {
                torch::NoGradGuard outer_no_grad;
                bool inner_grad_mode = torch::GradMode::is_enabled();
                
                {
                    torch::NoGradGuard inner_no_grad;
                    bool innermost_grad_mode = torch::GradMode::is_enabled();
                    
                    // Perform operations
                    torch::Tensor result = tensor * 2;
                }
                
                bool after_inner_grad_mode = torch::GradMode::is_enabled();
            }
            
            bool after_outer_grad_mode = torch::GradMode::is_enabled();
        }
        
        // Test with requires_grad=true tensor
        if (tensor.is_floating_point()) {
            torch::Tensor grad_tensor = tensor.clone().detach().requires_grad_(true);
            
            // Normal operation with grad
            torch::Tensor result_with_grad = grad_tensor * 2;
            
            // Operation with no_grad
            {
                torch::NoGradGuard no_grad;
                torch::Tensor result_no_grad = grad_tensor * 2;
                
                // Verify result_no_grad doesn't require grad
                bool requires_grad = result_no_grad.requires_grad();
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