#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least enough data for basic parameters
        if (Size < 16) {
            return 0;
        }

        // Extract upscale factor (1-8 to avoid memory issues)
        int upscale_factor = (Data[offset] % 8) + 1;
        offset += 1;

        // Extract tensor dimensions
        int batch_size = (Data[offset] % 4) + 1;  // 1-4
        offset += 1;
        
        int channels = upscale_factor * upscale_factor * ((Data[offset] % 4) + 1);  // Must be divisible by upscale_factor^2
        offset += 1;
        
        int height = (Data[offset] % 16) + 1;  // 1-16
        offset += 1;
        
        int width = (Data[offset] % 16) + 1;   // 1-16
        offset += 1;

        // Extract dtype (limit to common types)
        torch::ScalarType dtype;
        switch (Data[offset] % 4) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            default: dtype = torch::kInt64; break;
        }
        offset += 1;

        // Create input tensor with shape [N, C, H, W]
        auto input = torch::randn({batch_size, channels, height, width}, torch::dtype(dtype));
        
        // Test with random values if we have enough data
        if (offset + input.numel() * sizeof(float) <= Size) {
            auto data_ptr = input.data_ptr<float>();
            for (int64_t i = 0; i < std::min((int64_t)(Size - offset) / sizeof(float), input.numel()); ++i) {
                float val;
                memcpy(&val, Data + offset + i * sizeof(float), sizeof(float));
                if (std::isfinite(val)) {
                    data_ptr[i] = val;
                }
            }
        }

        // Test pixel_shuffle operation
        auto result = torch::pixel_shuffle(input, upscale_factor);

        // Verify output shape
        auto expected_shape = std::vector<int64_t>{
            batch_size, 
            channels / (upscale_factor * upscale_factor), 
            height * upscale_factor, 
            width * upscale_factor
        };
        
        if (result.sizes().vec() != expected_shape) {
            std::cout << "Shape mismatch in pixel_shuffle result" << std::endl;
            return -1;
        }

        // Test edge cases
        if (offset + 1 < Size) {
            // Test with different upscale factors
            for (int uf = 1; uf <= 3; ++uf) {
                if (channels % (uf * uf) == 0) {
                    auto edge_result = torch::pixel_shuffle(input, uf);
                    // Basic sanity check
                    if (edge_result.numel() != input.numel()) {
                        std::cout << "Numel mismatch in pixel_shuffle" << std::endl;
                        return -1;
                    }
                }
            }
        }

        // Test with zero tensor
        auto zero_input = torch::zeros_like(input);
        auto zero_result = torch::pixel_shuffle(zero_input, upscale_factor);

        // Test with ones tensor
        auto ones_input = torch::ones_like(input);
        auto ones_result = torch::pixel_shuffle(ones_input, upscale_factor);

        // Test gradient computation if input requires grad
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            auto grad_input = input.clone().requires_grad_(true);
            auto grad_output = torch::pixel_shuffle(grad_input, upscale_factor);
            auto loss = grad_output.sum();
            loss.backward();
            
            // Check that gradients exist
            if (!grad_input.grad().defined()) {
                std::cout << "Gradients not computed for pixel_shuffle" << std::endl;
                return -1;
            }
        }

        // Test with minimum valid tensor (1x(upscale_factor^2)x1x1)
        auto min_input = torch::randn({1, upscale_factor * upscale_factor, 1, 1}, torch::dtype(dtype));
        auto min_result = torch::pixel_shuffle(min_input, upscale_factor);

        // Verify minimum case output shape
        std::vector<int64_t> min_expected_shape = {1, 1, upscale_factor, upscale_factor};
        if (min_result.sizes().vec() != min_expected_shape) {
            std::cout << "Shape mismatch in minimum pixel_shuffle case" << std::endl;
            return -1;
        }

        // Test contiguity
        if (!result.is_contiguous()) {
            result = result.contiguous();
        }

        // Access some elements to trigger potential memory issues
        if (result.numel() > 0) {
            auto accessor = result.accessor<float, 4>();
            volatile float val = accessor[0][0][0][0];
            (void)val; // Suppress unused variable warning
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}