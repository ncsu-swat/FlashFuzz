#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with forward_with_indices result
#include <algorithm>      // For std::min, std::max

// --- Fuzzer Entry Point ---
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
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // AdaptiveMaxPool3d requires 4D (C, D, H, W) or 5D (N, C, D, H, W) input
        // Reshape to 5D with reasonable dimensions
        try {
            int64_t total_elements = input.numel();
            if (total_elements == 0) {
                return 0;
            }
            
            if (input.dim() == 5) {
                // Already 5D, ensure positive dimensions
                for (int i = 0; i < 5; i++) {
                    if (input.size(i) <= 0) {
                        return 0;
                    }
                }
            } else if (input.dim() == 4) {
                // 4D is also valid for AdaptiveMaxPool3d
                for (int i = 0; i < 4; i++) {
                    if (input.size(i) <= 0) {
                        return 0;
                    }
                }
            } else {
                // Reshape to 5D: (1, 1, D, H, W)
                // Distribute elements across spatial dimensions
                int64_t spatial_elements = total_elements;
                int64_t d = std::max(int64_t(1), static_cast<int64_t>(std::cbrt(spatial_elements)));
                int64_t remaining = spatial_elements / d;
                int64_t h = std::max(int64_t(1), static_cast<int64_t>(std::sqrt(remaining)));
                int64_t w = std::max(int64_t(1), remaining / h);
                
                // Adjust to match total elements
                int64_t product = d * h * w;
                if (product != total_elements) {
                    // Flatten and use as a single spatial dimension
                    d = total_elements;
                    h = 1;
                    w = 1;
                }
                
                input = input.reshape({1, 1, d, h, w});
            }
            
            // Ensure input is float type for pooling
            if (!input.is_floating_point()) {
                input = input.to(torch::kFloat32);
            }
        } catch (...) {
            // Reshape failed, skip this input
            return 0;
        }
        
        // Parse output size parameters from the remaining data
        // AdaptiveMaxPool3d needs exactly 3 output dimensions (D, H, W)
        int64_t out_d = 1, out_h = 1, out_w = 1;
        
        if (offset < Size) {
            out_d = std::max(int64_t(1), static_cast<int64_t>((Data[offset++] % 8) + 1));
        }
        if (offset < Size) {
            out_h = std::max(int64_t(1), static_cast<int64_t>((Data[offset++] % 8) + 1));
        }
        if (offset < Size) {
            out_w = std::max(int64_t(1), static_cast<int64_t>((Data[offset++] % 8) + 1));
        }
        
        // Create AdaptiveMaxPool3d module with 3D output size
        torch::nn::AdaptiveMaxPool3d pool(
            torch::nn::AdaptiveMaxPool3dOptions({out_d, out_h, out_w})
        );
        
        // Apply the pooling operation
        try {
            auto output = pool->forward(input);
            
            // Verify output shape
            (void)output.numel();
        } catch (...) {
            // Shape mismatch or other expected failure, skip silently
            return 0;
        }
        
        // Try forward_with_indices variant
        if (offset < Size && Data[offset] % 2 == 0) {
            try {
                auto result_tuple = pool->forward_with_indices(input);
                auto result = std::get<0>(result_tuple);
                auto indices = std::get<1>(result_tuple);
                
                // Use the results to prevent optimization
                (void)result.numel();
                (void)indices.numel();
            } catch (...) {
                // Expected failure with indices, skip silently
            }
        }
        
        // Test with different output size configurations
        if (offset + 1 < Size) {
            // Test with single value (applies to all dimensions)
            int64_t single_size = std::max(int64_t(1), static_cast<int64_t>((Data[offset] % 5) + 1));
            try {
                // Use brace initialization to avoid most vexing parse
                torch::nn::AdaptiveMaxPool3d pool_single{
                    torch::nn::AdaptiveMaxPool3dOptions(single_size)
                };
                auto output_single = pool_single->forward(input);
                (void)output_single.numel();
            } catch (...) {
                // Expected failure, skip silently
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}