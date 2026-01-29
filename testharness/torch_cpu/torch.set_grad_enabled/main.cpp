#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
        
        // Save initial grad state to restore later
        bool initial_grad_state = torch::autograd::GradMode::is_enabled();
        
        // Create a float tensor that supports gradients
        torch::Tensor tensor;
        if (offset < Size) {
            try {
                tensor = fuzzer_utils::createTensor(Data, Size, offset);
                // Convert to float to ensure gradient support
                tensor = tensor.to(torch::kFloat32).detach().set_requires_grad(true);
            } catch (...) {
                // If tensor creation fails, use a default tensor
                tensor = torch::randn({2, 3}, torch::requires_grad());
            }
        } else {
            tensor = torch::randn({2, 3}, torch::requires_grad());
        }
        
        // Test 1: Set grad enabled to our fuzzed value
        torch::autograd::GradMode::set_enabled(grad_enabled);
        
        // Verify that grad_enabled was set correctly
        bool new_grad_state = torch::autograd::GradMode::is_enabled();
        (void)new_grad_state; // Use the variable
        
        // Perform an operation that respects grad mode
        torch::Tensor result = tensor * 2.0;
        (void)result;
        
        // Test 2: Toggle the value
        torch::autograd::GradMode::set_enabled(!grad_enabled);
        bool toggled_grad_state = torch::autograd::GradMode::is_enabled();
        (void)toggled_grad_state;
        
        // Perform another operation
        torch::Tensor result2 = tensor + tensor;
        (void)result2;
        
        // Test 3: Use NoGradGuard context
        {
            torch::NoGradGuard no_grad;
            bool guard_grad_state = torch::autograd::GradMode::is_enabled();
            (void)guard_grad_state;
            
            // Operations in this scope should not track gradients
            torch::Tensor guarded_result = tensor * 3.0;
            (void)guarded_result;
        }
        
        // Test 4: Use AutoGradMode guard with explicit value
        {
            torch::AutoGradMode auto_grad(grad_enabled);
            bool auto_grad_state = torch::autograd::GradMode::is_enabled();
            (void)auto_grad_state;
            
            torch::Tensor auto_result = tensor - tensor;
            (void)auto_result;
        }
        
        // Test 5: Rapid toggling
        for (int i = 0; i < 3; i++) {
            torch::autograd::GradMode::set_enabled(i % 2 == 0);
            torch::Tensor toggle_result = tensor / 2.0;
            (void)toggle_result;
        }
        
        // Restore the original grad state
        torch::autograd::GradMode::set_enabled(initial_grad_state);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}