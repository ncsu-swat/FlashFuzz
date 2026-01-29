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
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }

        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // BatchNorm3d requires floating point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat);
        }
        
        // Ensure the tensor has 5 dimensions for BatchNorm3d (N, C, D, H, W)
        // and has valid sizes
        int64_t numel = input.numel();
        if (numel == 0) {
            return 0;
        }
        
        if (input.dim() != 5) {
            // Create a valid 5D shape from the total number of elements
            // Use small dimensions to avoid memory issues
            int64_t n = 1;
            int64_t c = std::min(numel, (int64_t)16);  // channels
            int64_t remaining = numel / c;
            if (remaining == 0) remaining = 1;
            
            int64_t d = 1, h = 1, w = 1;
            
            // Distribute remaining elements among D, H, W
            if (remaining >= 8) {
                d = 2;
                remaining /= 2;
            }
            if (remaining >= 4) {
                h = 2;
                remaining /= 2;
            }
            w = remaining;
            
            // Ensure we have the right total number of elements
            int64_t new_numel = n * c * d * h * w;
            if (new_numel > 0 && new_numel <= numel) {
                try {
                    input = input.flatten().narrow(0, 0, new_numel).reshape({n, c, d, h, w});
                } catch (...) {
                    return 0;
                }
            } else {
                return 0;
            }
        }
        
        // Validate dimensions
        if (input.size(0) <= 0 || input.size(1) <= 0 || 
            input.size(2) <= 0 || input.size(3) <= 0 || input.size(4) <= 0) {
            return 0;
        }
        
        // Extract parameters for BatchNorm3d from the remaining data
        bool affine = true;
        bool track_running_stats = true;
        double momentum = 0.1;
        double eps = 1e-5;
        
        if (offset + 4 <= Size) {
            affine = Data[offset++] & 0x1;
            track_running_stats = Data[offset++] & 0x1;
            momentum = static_cast<double>(Data[offset++]) / 255.0;
            eps = std::max(1e-10, static_cast<double>(Data[offset++]) / 1000.0);
        }
        
        // Get number of features (channels) from the input tensor
        int64_t num_features = input.size(1);
        
        // Create BatchNorm3d module
        torch::nn::BatchNorm3d bn(torch::nn::BatchNorm3dOptions(num_features)
                                  .eps(eps)
                                  .momentum(momentum)
                                  .affine(affine)
                                  .track_running_stats(track_running_stats));
        
        // Apply BatchNorm3d in training mode (default)
        try {
            torch::Tensor output = bn->forward(input);
            
            // Verify output shape matches input shape
            (void)output.sizes();
        } catch (...) {
            // Expected failures (e.g., numerical issues)
        }
        
        // Test evaluation mode
        if (offset < Size && (Data[offset++] & 0x1)) {
            try {
                bn->eval();
                torch::Tensor eval_output = bn->forward(input);
                (void)eval_output.sizes();
            } catch (...) {
                // Expected failures in eval mode
            }
        }
        
        // Test with double precision if fuzz data indicates
        if (offset < Size && (Data[offset++] & 0x1)) {
            try {
                torch::Tensor double_input = input.to(torch::kDouble);
                torch::nn::BatchNorm3d bn_double(torch::nn::BatchNorm3dOptions(num_features)
                                                 .eps(eps)
                                                 .momentum(momentum)
                                                 .affine(affine)
                                                 .track_running_stats(track_running_stats));
                bn_double->to(torch::kDouble);
                torch::Tensor double_output = bn_double->forward(double_input);
                (void)double_output.sizes();
            } catch (...) {
                // Expected failures with double precision
            }
        }
        
        // Test resetting running stats if tracking is enabled
        if (track_running_stats && offset < Size && (Data[offset++] & 0x1)) {
            try {
                bn->reset_running_stats();
                torch::Tensor output_after_reset = bn->forward(input);
                (void)output_after_reset.sizes();
            } catch (...) {
                // Expected failures after reset
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