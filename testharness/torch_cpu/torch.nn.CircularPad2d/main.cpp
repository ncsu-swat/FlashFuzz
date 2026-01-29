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
        // Need enough bytes for tensor creation and padding values
        if (Size < 12) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract padding values first (use modulo to keep them reasonable)
        int64_t left = static_cast<int64_t>(Data[offset++] % 16);
        int64_t right = static_cast<int64_t>(Data[offset++] % 16);
        int64_t top = static_cast<int64_t>(Data[offset++] % 16);
        int64_t bottom = static_cast<int64_t>(Data[offset++] % 16);
        
        // Extract shape parameters for a proper 4D tensor (N, C, H, W)
        int64_t batch = 1 + (Data[offset++] % 4);      // 1-4
        int64_t channels = 1 + (Data[offset++] % 4);   // 1-4
        int64_t height = 1 + (Data[offset++] % 32);    // 1-32
        int64_t width = 1 + (Data[offset++] % 32);     // 1-32
        
        // Circular padding requires padding < corresponding dimension
        // Adjust padding if necessary
        if (left >= width) left = width - 1;
        if (right >= width) right = width - 1;
        if (top >= height) top = height - 1;
        if (bottom >= height) bottom = height - 1;
        
        // Ensure at least some padding is applied for coverage
        if (left == 0 && right == 0 && top == 0 && bottom == 0) {
            left = 1;
            if (width <= 1) width = 2;
        }
        
        // Create a 4D tensor suitable for 2D operations
        torch::Tensor input_tensor = torch::randn({batch, channels, height, width});
        
        // Also test with tensor created from fuzzer data for dtype coverage
        if (Size > offset + 4) {
            try {
                torch::Tensor fuzz_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
                
                // Reshape to 4D if possible
                auto numel = fuzz_tensor.numel();
                if (numel >= 4) {
                    int64_t h = 2;
                    int64_t w = numel / 2;
                    if (w > 0) {
                        fuzz_tensor = fuzz_tensor.reshape({1, 1, h, w});
                        
                        // Adjust padding for fuzz tensor dimensions
                        int64_t fl = left % h;
                        int64_t fr = right % h;
                        int64_t ft = top % w;
                        int64_t fb = bottom % w;
                        if (fl == 0 && fr == 0 && ft == 0 && fb == 0) fl = 1;
                        
                        // Only apply padding to float types
                        if (fuzz_tensor.is_floating_point()) {
                            torch::Tensor fuzz_output = torch::nn::functional::pad(
                                fuzz_tensor,
                                torch::nn::functional::PadFuncOptions({fl, fr, ft, fb})
                                    .mode(torch::kCircular));
                            (void)fuzz_output.sizes();
                        }
                    }
                }
            } catch (...) {
                // Silently ignore errors from fuzz tensor operations
            }
        }
        
        // Apply circular padding using torch::nn::functional::pad
        torch::Tensor output = torch::nn::functional::pad(
            input_tensor,
            torch::nn::functional::PadFuncOptions({left, right, top, bottom})
                .mode(torch::kCircular));
        
        // Force evaluation
        (void)output.sizes();
        (void)output.sum().item<float>();
        
        // Also test with 3D tensor (C, H, W)
        torch::Tensor input_3d = torch::randn({channels, height, width});
        torch::Tensor output_3d = torch::nn::functional::pad(
            input_3d,
            torch::nn::functional::PadFuncOptions({left, right, top, bottom})
                .mode(torch::kCircular));
        (void)output_3d.sizes();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}