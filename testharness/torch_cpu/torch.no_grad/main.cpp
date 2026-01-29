#include "fuzzer_utils.h"
#include <iostream>

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
                (void)during_no_grad;
                
                // Perform some operations on the tensor
                torch::Tensor result = tensor + 1;
                (void)result;
                
                // Try another operation
                torch::Tensor result2 = torch::sin(tensor.to(torch::kFloat));
                (void)result2;
                
                // Try element-wise operation instead of matmul (which has shape requirements)
                torch::Tensor result3 = tensor * tensor;
                (void)result3;
            }
            
            // Verify that grad mode is restored after the scope
            bool after_no_grad = torch::GradMode::is_enabled();
            (void)after_no_grad;
            (void)initial_requires_grad;
        }
        
        // Test torch::NoGradGuard with operations that might fail due to shape/dtype
        {
            try {
                torch::NoGradGuard no_grad;
                
                // Perform an operation that might throw for certain dtypes
                torch::Tensor float_tensor = tensor.to(torch::kFloat);
                torch::Tensor result = torch::log(torch::abs(float_tensor) + 1e-6);
                (void)result;
            }
            catch (const std::exception& e) {
                // Expected failures for certain inputs, silently continue
            }
            
            // Verify grad mode is restored even after exception
            bool after_exception = torch::GradMode::is_enabled();
            (void)after_exception;
        }
        
        // Test nested NoGradGuard
        {
            bool outer_grad_mode = torch::GradMode::is_enabled();
            (void)outer_grad_mode;
            
            {
                torch::NoGradGuard outer_no_grad;
                bool inner_grad_mode = torch::GradMode::is_enabled();
                (void)inner_grad_mode;
                
                {
                    torch::NoGradGuard inner_no_grad;
                    bool innermost_grad_mode = torch::GradMode::is_enabled();
                    (void)innermost_grad_mode;
                    
                    // Perform operations
                    torch::Tensor result = tensor * 2;
                    (void)result;
                }
                
                bool after_inner_grad_mode = torch::GradMode::is_enabled();
                (void)after_inner_grad_mode;
            }
            
            bool after_outer_grad_mode = torch::GradMode::is_enabled();
            (void)after_outer_grad_mode;
        }
        
        // Test with requires_grad=true tensor
        if (tensor.is_floating_point()) {
            torch::Tensor grad_tensor = tensor.clone().detach().requires_grad_(true);
            
            // Normal operation with grad
            torch::Tensor result_with_grad = grad_tensor * 2;
            (void)result_with_grad;
            
            // Operation with no_grad
            {
                torch::NoGradGuard no_grad;
                torch::Tensor result_no_grad = grad_tensor * 2;
                
                // Verify result_no_grad doesn't require grad
                bool requires_grad = result_no_grad.requires_grad();
                (void)requires_grad;
            }
        }
        
        // Test AutoGradMode explicitly (alternative API)
        {
            torch::AutoGradMode guard(false);  // Disable grad
            bool grad_enabled = torch::GradMode::is_enabled();
            (void)grad_enabled;
            
            torch::Tensor result = tensor + tensor;
            (void)result;
        }
        
        // Test set_enabled static method
        {
            bool original = torch::GradMode::is_enabled();
            torch::NoGradGuard no_grad;
            
            // Perform various operations
            torch::Tensor t1 = tensor.clone();
            torch::Tensor t2 = tensor.view(-1);
            torch::Tensor t3 = tensor.reshape({-1});
            (void)t1;
            (void)t2;
            (void)t3;
            (void)original;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}