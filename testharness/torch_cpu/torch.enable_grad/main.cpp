#include "fuzzer_utils.h"
#include <iostream>

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

        // Create a tensor to work with
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            tensor = torch::randn({2, 3});
        }

        // Get control values from fuzzer input
        uint8_t mode_selector = 0;
        if (offset < Size) {
            mode_selector = Data[offset++];
        }

        // Save original grad mode state
        bool original_grad_mode = torch::GradMode::is_enabled();

        // Test different gradient control mechanisms based on fuzzer input
        switch (mode_selector % 4) {
            case 0: {
                // Test torch::GradMode::set_enabled(true)
                torch::GradMode::set_enabled(true);
                
                torch::Tensor x = tensor.clone().detach().to(torch::kFloat32).requires_grad_(true);
                torch::Tensor y = x * x + 2 * x;
                
                // Verify grad mode is enabled
                if (!torch::GradMode::is_enabled()) {
                    break;
                }
                
                // Compute gradients
                torch::Tensor grad_output = torch::ones_like(y);
                try {
                    y.backward(grad_output);
                } catch (...) {
                    // May fail for certain tensor types/shapes
                }
                break;
            }
            case 1: {
                // Test torch::GradMode::set_enabled(false)
                torch::GradMode::set_enabled(false);
                
                // Operations with grad mode disabled
                torch::Tensor x = tensor.clone().detach().to(torch::kFloat32);
                torch::Tensor y = x * x + 2 * x;
                
                // Verify grad mode is disabled
                if (torch::GradMode::is_enabled()) {
                    break;
                }
                
                // y should not have grad_fn when grad mode is disabled
                // This is expected behavior
                break;
            }
            case 2: {
                // Test torch::NoGradGuard (RAII-style gradient disabling)
                {
                    torch::NoGradGuard no_grad;
                    
                    // Inside this scope, grad should be disabled
                    torch::Tensor x = tensor.clone().detach().to(torch::kFloat32);
                    torch::Tensor y = x * x + 2 * x;
                    
                    // Operations here won't track gradients
                    (void)y; // Suppress unused variable warning
                }
                // Outside the scope, grad mode should be restored
                break;
            }
            case 3: {
                // Test torch::AutoGradMode (RAII-style gradient control)
                bool enable_grad = (offset < Size) ? (Data[offset++] & 0x1) : true;
                
                {
                    torch::AutoGradMode auto_grad_mode(enable_grad);
                    
                    torch::Tensor x = tensor.clone().detach().to(torch::kFloat32);
                    if (enable_grad) {
                        x = x.requires_grad_(true);
                    }
                    torch::Tensor y = x * x + 2 * x;
                    
                    if (enable_grad && torch::GradMode::is_enabled()) {
                        try {
                            torch::Tensor grad_output = torch::ones_like(y);
                            y.backward(grad_output);
                        } catch (...) {
                            // May fail for certain tensor configurations
                        }
                    }
                }
                // Grad mode restored after scope ends
                break;
            }
        }

        // Test toggling grad mode multiple times
        uint8_t toggle_count = 0;
        if (offset < Size) {
            toggle_count = Data[offset++] % 8; // Limit to reasonable number
        }
        
        for (uint8_t i = 0; i < toggle_count; i++) {
            bool new_mode = (offset < Size) ? (Data[offset++] & 0x1) : (i % 2 == 0);
            torch::GradMode::set_enabled(new_mode);
            
            // Quick operation to verify state
            torch::Tensor t = torch::ones({2, 2}, torch::kFloat32);
            if (new_mode) {
                t = t.requires_grad_(true);
            }
            torch::Tensor result = t * 2;
            (void)result;
        }

        // Restore original grad mode
        torch::GradMode::set_enabled(original_grad_mode);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}