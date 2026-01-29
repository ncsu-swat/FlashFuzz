#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        // Need enough bytes for parameters
        if (Size < 32) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse size parameter (must be positive odd number typically)
        int64_t size_raw;
        std::memcpy(&size_raw, Data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        int64_t size = (std::abs(size_raw) % 10) + 1; // 1-10 range
        
        // Parse alpha parameter
        double alpha_raw;
        std::memcpy(&alpha_raw, Data + offset, sizeof(double));
        offset += sizeof(double);
        // Clamp alpha to reasonable range to avoid NaN/Inf
        double alpha = std::isfinite(alpha_raw) ? std::fmod(std::abs(alpha_raw), 1.0) + 1e-6 : 1e-4;
        
        // Parse beta parameter
        double beta_raw;
        std::memcpy(&beta_raw, Data + offset, sizeof(double));
        offset += sizeof(double);
        // Clamp beta to reasonable range
        double beta = std::isfinite(beta_raw) ? std::fmod(std::abs(beta_raw), 2.0) + 0.1 : 0.75;
        
        // Parse k parameter
        double k_raw;
        std::memcpy(&k_raw, Data + offset, sizeof(double));
        offset += sizeof(double);
        // k should be positive to avoid division issues
        double k = std::isfinite(k_raw) ? std::fmod(std::abs(k_raw), 10.0) + 0.1 : 1.0;
        
        // Create LocalResponseNorm module
        torch::nn::LocalResponseNorm lrn(
            torch::nn::LocalResponseNormOptions(size)
                .alpha(alpha)
                .beta(beta)
                .k(k)
        );
        
        // LocalResponseNorm expects input of shape (N, C, ...) with at least 3 dimensions
        // Create appropriate input tensor
        
        // Parse batch and channel dimensions from fuzzer data
        uint8_t batch_byte = (offset < Size) ? Data[offset++] : 1;
        uint8_t channel_byte = (offset < Size) ? Data[offset++] : 4;
        uint8_t spatial_byte = (offset < Size) ? Data[offset++] : 8;
        
        int64_t batch = (batch_byte % 4) + 1;      // 1-4
        int64_t channels = (channel_byte % 16) + 1; // 1-16
        int64_t spatial = (spatial_byte % 16) + 1;  // 1-16
        
        // Test with 3D input (N, C, L)
        {
            torch::Tensor input_3d = torch::randn({batch, channels, spatial});
            try {
                torch::Tensor output = lrn->forward(input_3d);
                torch::Tensor sum = output.sum();
                (void)sum;
            } catch (const std::exception&) {
                // Shape or parameter issues - silently ignore
            }
        }
        
        // Test with 4D input (N, C, H, W) - common for images
        {
            int64_t height = (spatial_byte % 8) + 1;
            int64_t width = ((spatial_byte >> 4) % 8) + 1;
            torch::Tensor input_4d = torch::randn({batch, channels, height, width});
            try {
                torch::Tensor output = lrn->forward(input_4d);
                torch::Tensor sum = output.sum();
                (void)sum;
            } catch (const std::exception&) {
                // Shape or parameter issues - silently ignore
            }
        }
        
        // Test with 5D input (N, C, D, H, W) - common for 3D convolutions
        {
            int64_t depth = (spatial_byte % 4) + 1;
            int64_t height = ((spatial_byte >> 2) % 4) + 1;
            int64_t width = ((spatial_byte >> 4) % 4) + 1;
            torch::Tensor input_5d = torch::randn({batch, channels, depth, height, width});
            try {
                torch::Tensor output = lrn->forward(input_5d);
                torch::Tensor sum = output.sum();
                (void)sum;
            } catch (const std::exception&) {
                // Shape or parameter issues - silently ignore
            }
        }
        
        // Test with tensor created from fuzzer data
        torch::Tensor input_fuzz = fuzzer_utils::createTensor(Data, Size, offset);
        if (input_fuzz.dim() >= 3) {
            try {
                torch::Tensor output = lrn->forward(input_fuzz);
                torch::Tensor sum = output.sum();
                (void)sum;
            } catch (const std::exception&) {
                // Shape mismatch or other expected issues
            }
        }
        
        // Test with different dtypes
        {
            torch::Tensor input_double = torch::randn({batch, channels, spatial}, torch::kDouble);
            try {
                torch::Tensor output = lrn->forward(input_double);
                (void)output.sum();
            } catch (const std::exception&) {
                // dtype issues - silently ignore
            }
        }
        
        // Test with edge case: size larger than channels
        if (size > channels) {
            try {
                torch::Tensor input = torch::randn({batch, channels, spatial});
                torch::Tensor output = lrn->forward(input);
                (void)output.sum();
            } catch (const std::exception&) {
                // Expected edge case
            }
        }
        
        // Test with extreme but valid parameters
        try {
            torch::nn::LocalResponseNorm lrn_edge(
                torch::nn::LocalResponseNormOptions(1)
                    .alpha(1e-8)
                    .beta(0.01)
                    .k(0.001)
            );
            torch::Tensor input = torch::randn({1, 4, 4});
            torch::Tensor output = lrn_edge->forward(input);
            (void)output.sum();
        } catch (const std::exception&) {
            // Edge case handling
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}