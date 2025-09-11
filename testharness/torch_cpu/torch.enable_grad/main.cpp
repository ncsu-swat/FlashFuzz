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
        
        // Create a tensor to work with
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            tensor = torch::randn({2, 3});
        }
        
        // Test enable_grad functionality
        bool was_grad_enabled = torch::GradMode::is_enabled();
        
        // Get a boolean from the input data if available
        bool enable_value = true;
        if (offset < Size) {
            enable_value = Data[offset++] & 0x1;
        }
        
        // Enable or disable grad based on the input
        if (enable_value) {
            torch::GradMode::set_enabled(true);
        } else {
            torch::GradMode::set_enabled(false);
        }
        
        // Verify that grad mode was set correctly
        bool is_grad_enabled = torch::GradMode::is_enabled();
        
        // Create a tensor that requires grad
        torch::Tensor x = tensor.clone().detach().requires_grad_(true);
        
        // Perform some operations that would normally accumulate gradients
        torch::Tensor y = x * x + 2 * x;
        
        // If grad is enabled, we should be able to compute gradients
        if (is_grad_enabled) {
            // Try to compute gradients
            if (y.dim() > 0) {
                torch::Tensor grad_output = torch::ones_like(y);
                y.backward(grad_output);
                
                // Check if gradients were computed
                if (!x.grad().defined()) {
                    throw std::runtime_error("Gradients not computed when grad mode was enabled");
                }
            } else {
                // For scalar outputs
                y.backward();
                
                // Check if gradients were computed
                if (!x.grad().defined()) {
                    throw std::runtime_error("Gradients not computed when grad mode was enabled");
                }
            }
        } else {
            // When grad is disabled, backward should throw an exception
            try {
                if (y.dim() > 0) {
                    torch::Tensor grad_output = torch::ones_like(y);
                    y.backward(grad_output);
                } else {
                    y.backward();
                }
                throw std::runtime_error("Backward did not throw when grad mode was disabled");
            } catch (const c10::Error&) {
                // Expected behavior when grad is disabled
            }
        }
        
        // Restore previous grad mode
        torch::GradMode::set_enabled(was_grad_enabled);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
