#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for the grad_enabled flag
        if (Size < 1) {
            return 0;
        }
        
        // Extract a boolean value from the first byte to determine if grad should be enabled
        bool grad_enabled = (Data[0] % 2 == 1);
        offset++;
        
        // Create a tensor with requires_grad=true to test grad_enabled functionality
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
            tensor = tensor.set_requires_grad(true);
        } else {
            // If we don't have enough data for a tensor, create a simple one
            tensor = torch::randn({2, 3}, torch::requires_grad());
        }
        
        // Get initial grad enabled state
        bool initial_grad_state = torch::autograd::GradMode::is_enabled();
        
        // Set grad enabled to our test value
        torch::autograd::GradMode::set_enabled(grad_enabled);
        
        // Verify that grad_enabled was set correctly
        bool new_grad_state = torch::autograd::GradMode::is_enabled();
        if (new_grad_state != grad_enabled) {
            throw std::runtime_error("torch::autograd::GradMode::set_enabled failed to set the correct state");
        }
        
        // Test that operations respect the grad_enabled setting
        torch::Tensor result = tensor * 2.0;
        
        // If grad is disabled, result should not require grad even if input does
        if (!grad_enabled && result.requires_grad()) {
            throw std::runtime_error("Gradient tracking should be disabled but tensor requires_grad is true");
        }
        
        // If grad is enabled, result should require grad if input does
        if (grad_enabled && !result.requires_grad()) {
            throw std::runtime_error("Gradient tracking should be enabled but tensor requires_grad is false");
        }
        
        // Test with a different value to ensure toggle works
        torch::autograd::GradMode::set_enabled(!grad_enabled);
        bool toggled_grad_state = torch::autograd::GradMode::is_enabled();
        if (toggled_grad_state != !grad_enabled) {
            throw std::runtime_error("torch::autograd::GradMode::set_enabled failed to toggle the state");
        }
        
        // Test with a context manager (GradMode guard)
        {
            torch::NoGradGuard no_grad;
            bool guard_grad_state = torch::autograd::GradMode::is_enabled();
            if (guard_grad_state) {
                throw std::runtime_error("NoGradGuard failed to disable grad");
            }
            
            // Operations in this scope should not track gradients
            torch::Tensor guarded_result = tensor * 3.0;
            if (guarded_result.requires_grad()) {
                throw std::runtime_error("NoGradGuard should prevent gradient tracking");
            }
        }
        
        // Grad state should be restored after guard goes out of scope
        bool restored_grad_state = torch::autograd::GradMode::is_enabled();
        if (restored_grad_state != !grad_enabled) {
            throw std::runtime_error("Grad state not properly restored after guard");
        }
        
        // Restore the original grad state
        torch::autograd::GradMode::set_enabled(initial_grad_state);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}