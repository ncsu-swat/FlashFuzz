#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Check if we have enough data to proceed
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
        
        // Create a tensor with requires_grad=true to test gradient tracking
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Set requires_grad on the tensor
            tensor = tensor.set_requires_grad(true);
            
            // Verify that requires_grad is properly set based on grad_enabled state
            bool tensor_requires_grad = tensor.requires_grad();
            
            // Perform some operation on the tensor
            torch::Tensor result = tensor.sin();
            
            // Check if the result has requires_grad set correctly
            bool result_requires_grad = result.requires_grad();
            
            // Test gradient computation if gradients are enabled
            if (current_grad_state && tensor_requires_grad) {
                // Sum to get a scalar output for backward
                torch::Tensor sum = result.sum();
                sum.backward();
                
                // Check if gradient was computed
                bool has_grad = tensor.grad().defined();
            }
        }
        
        // Test gradient context manager
        {
            torch::NoGradGuard no_grad;
            bool grad_disabled = !torch::autograd::GradMode::is_enabled();
            
            // Create another tensor inside NoGradGuard
            if (offset < Size) {
                torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                tensor2 = tensor2.set_requires_grad(true);
                
                // Even with requires_grad=true, operations should not track gradients
                torch::Tensor result2 = tensor2.cos();
                bool result2_requires_grad = result2.requires_grad();
            }
        }
        
        // Check that grad state is restored after the guard is destroyed
        bool restored_grad_state = torch::autograd::GradMode::is_enabled();
        
        // Test with GradMode guard
        {
            torch::AutoGradMode grad_mode(enable_grad);
            bool grad_mode_state = torch::autograd::GradMode::is_enabled();
            
            if (offset < Size) {
                torch::Tensor tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
                tensor3 = tensor3.set_requires_grad(true);
                torch::Tensor result3 = tensor3.exp();
            }
        }
        
        // Restore the original gradient state
        torch::autograd::GradMode::set_enabled(initial_grad_state);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}