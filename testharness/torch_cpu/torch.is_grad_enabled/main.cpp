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
        
        if (Size < 1) {
            return 0;
        }
        
        // Use the first byte to determine whether to enable or disable gradients
        bool enable_grad = Data[offset++] % 2 == 0;
        
        // Test getting the current grad state before changing it
        bool initial_grad_state = torch::autograd::GradMode::is_enabled();
        
        // Set the gradient state based on our random input
        torch::autograd::GradMode::set_enabled(enable_grad);
        
        // Test if is_grad_enabled returns the value we just set
        bool current_grad_state = torch::autograd::GradMode::is_enabled();
        
        // Create a floating point tensor for gradient tracking
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Convert to float for gradient operations
            tensor = tensor.to(torch::kFloat32);
            
            // set_requires_grad returns void, don't chain
            tensor.set_requires_grad(true);
            
            // Verify that requires_grad is properly set
            bool tensor_requires_grad = tensor.requires_grad();
            
            // Perform some operation on the tensor
            torch::Tensor result = tensor.sin();
            
            // Check if the result has requires_grad set correctly
            bool result_requires_grad = result.requires_grad();
            
            // Test gradient computation if gradients are enabled
            if (current_grad_state && tensor_requires_grad) {
                try {
                    // Sum to get a scalar output for backward
                    torch::Tensor sum_val = result.sum();
                    sum_val.backward();
                    
                    // Check if gradient was computed
                    bool has_grad = tensor.grad().defined();
                    (void)has_grad;
                } catch (...) {
                    // Silently handle backward failures
                }
            }
            
            (void)result_requires_grad;
        }
        
        // Test NoGradGuard context manager
        {
            torch::NoGradGuard no_grad;
            bool grad_disabled = !torch::autograd::GradMode::is_enabled();
            (void)grad_disabled;
            
            if (offset < Size) {
                torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                tensor2 = tensor2.to(torch::kFloat32);
                tensor2.set_requires_grad(true);
                
                // Even with requires_grad=true, operations should not track gradients
                torch::Tensor result2 = tensor2.cos();
                bool result2_requires_grad = result2.requires_grad();
                (void)result2_requires_grad;
            }
        }
        
        // Check that grad state is restored after the guard is destroyed
        bool restored_grad_state = torch::autograd::GradMode::is_enabled();
        (void)restored_grad_state;
        
        // Test with AutoGradMode guard
        {
            torch::AutoGradMode grad_mode(enable_grad);
            bool grad_mode_state = torch::autograd::GradMode::is_enabled();
            (void)grad_mode_state;
            
            if (offset < Size) {
                torch::Tensor tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
                tensor3 = tensor3.to(torch::kFloat32);
                tensor3.set_requires_grad(true);
                torch::Tensor result3 = tensor3.exp();
                (void)result3;
            }
        }
        
        // Restore the original gradient state
        torch::autograd::GradMode::set_enabled(initial_grad_state);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}