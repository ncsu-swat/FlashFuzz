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
        // Need at least enough bytes for parameters and tensor creation
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse size parameter (must be positive odd number for proper LRN behavior)
        int64_t size = 1 + (Data[offset++] % 7); // Size between 1 and 7
        
        // Parse alpha parameter
        uint8_t alpha_byte = Data[offset++];
        float alpha = static_cast<float>(alpha_byte) / 255.0f * 0.001f + 0.0001f; // [0.0001, 0.0011]
        
        // Parse beta parameter
        uint8_t beta_byte = Data[offset++];
        float beta = static_cast<float>(beta_byte) / 255.0f + 0.5f; // [0.5, 1.5]
        
        // Parse k parameter
        uint8_t k_byte = Data[offset++];
        float k = static_cast<float>(k_byte) / 255.0f + 0.5f; // [0.5, 1.5]
        
        // Parse dimensions for creating proper input tensor
        // LocalResponseNorm requires at least 3D input: (N, C, ...)
        uint8_t batch_size = 1 + (Data[offset++] % 4);      // 1-4
        uint8_t num_channels = 1 + (Data[offset++] % 16);   // 1-16
        uint8_t spatial_dim = 1 + (Data[offset++] % 8);     // 1-8
        uint8_t extra_dim_flag = Data[offset++];
        
        // Create input tensor with appropriate shape for LocalResponseNorm
        torch::Tensor input;
        if (extra_dim_flag % 2 == 0) {
            // 3D input: (N, C, L)
            input = torch::randn({batch_size, num_channels, spatial_dim});
        } else {
            // 4D input: (N, C, H, W)
            uint8_t spatial_dim2 = (offset < Size) ? (1 + Data[offset++] % 8) : spatial_dim;
            input = torch::randn({batch_size, num_channels, spatial_dim, spatial_dim2});
        }
        
        // Create LocalResponseNorm module
        torch::nn::LocalResponseNorm lrn(
            torch::nn::LocalResponseNormOptions(size)
                .alpha(alpha)
                .beta(beta)
                .k(k)
        );
        
        // Apply the operation
        torch::Tensor output = lrn->forward(input);
        
        // Test with different configurations using remaining fuzz data
        if (offset + 4 < Size) {
            int64_t size2 = 1 + (Data[offset++] % 7);
            float alpha2 = static_cast<float>(Data[offset++]) / 255.0f * 0.001f + 0.0001f;
            float beta2 = static_cast<float>(Data[offset++]) / 255.0f + 0.5f;
            float k2 = static_cast<float>(Data[offset++]) / 255.0f + 0.5f;
            
            torch::nn::LocalResponseNorm lrn2(
                torch::nn::LocalResponseNormOptions(size2)
                    .alpha(alpha2)
                    .beta(beta2)
                    .k(k2)
            );
            
            // Inner try-catch for expected failures (shape issues)
            try {
                torch::Tensor output2 = lrn2->forward(input);
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Test with 5D input if there's more data
        if (offset + 2 < Size) {
            uint8_t d1 = 1 + (Data[offset++] % 4);
            uint8_t d2 = 1 + (Data[offset++] % 4);
            
            try {
                torch::Tensor input5d = torch::randn({batch_size, num_channels, spatial_dim, d1, d2});
                torch::Tensor output5d = lrn->forward(input5d);
            } catch (...) {
                // Silently ignore - 5D may not be supported
            }
        }
        
        // Test with different dtypes
        try {
            torch::Tensor input_double = input.to(torch::kDouble);
            torch::Tensor output_double = lrn->forward(input_double);
        } catch (...) {
            // Silently ignore dtype issues
        }
        
        // Test with edge case: single channel
        try {
            torch::Tensor single_channel = torch::randn({1, 1, spatial_dim});
            torch::Tensor output_single = lrn->forward(single_channel);
        } catch (...) {
            // Silently ignore
        }
        
        // Test with large size parameter relative to channels
        if (offset < Size) {
            try {
                int64_t large_size = num_channels + (Data[offset++] % 5);
                torch::nn::LocalResponseNorm lrn_large(
                    torch::nn::LocalResponseNormOptions(large_size)
                        .alpha(alpha)
                        .beta(beta)
                        .k(k)
                );
                torch::Tensor output_large = lrn_large->forward(input);
            } catch (...) {
                // Silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}