#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        
        // Need at least a few bytes to create a tensor and output size
        if (Size < 8) {
            return 0;
        }
        
        // Extract output size first
        int64_t output_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&output_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure output_size is within reasonable bounds (1 to 100)
            output_size = (std::abs(output_size) % 100) + 1;
        }
        
        // Create input tensor from remaining data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // AdaptiveAvgPool1d expects:
        // - 2D input: (channels, input_size) - unbatched
        // - 3D input: (batch_size, channels, input_size) - batched
        // Reshape input to be 3D if needed
        if (input.numel() == 0) {
            // Create a minimal valid tensor
            input = torch::randn({1, 1, 4});
        } else if (input.dim() == 0) {
            // Scalar - reshape to 3D
            input = input.view({1, 1, 1});
        } else if (input.dim() == 1) {
            // 1D - add batch and channel dims
            input = input.unsqueeze(0).unsqueeze(0);
        } else if (input.dim() == 2) {
            // 2D - could be (C, L) unbatched or need to add batch dim
            // Treat as unbatched (C, L)
            input = input.unsqueeze(0);
        } else if (input.dim() > 3) {
            // Flatten extra dimensions into batch
            auto sizes = input.sizes();
            int64_t batch = 1;
            for (int i = 0; i < input.dim() - 2; i++) {
                batch *= sizes[i];
            }
            input = input.view({batch, sizes[input.dim()-2], sizes[input.dim()-1]});
        }
        // Now input is 3D: (N, C, L)
        
        // Ensure input has valid dimensions
        if (input.size(-1) < 1) {
            return 0;
        }
        
        // Ensure input is float type (required for pooling)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create the AdaptiveAvgPool1d module
        torch::nn::AdaptiveAvgPool1d pool(output_size);
        
        // Apply the pooling operation
        try {
            torch::Tensor output = pool(input);
            // Use output to prevent optimization
            (void)output.numel();
        } catch (const c10::Error&) {
            // Expected for invalid configurations
        }
        
        // Try with different output sizes
        if (offset + sizeof(int64_t) <= Size) {
            int64_t output_size2;
            std::memcpy(&output_size2, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure output_size is within reasonable bounds
            output_size2 = (std::abs(output_size2) % 100) + 1;
            
            torch::nn::AdaptiveAvgPool1d pool2(output_size2);
            
            try {
                torch::Tensor output2 = pool2(input);
                (void)output2.numel();
            } catch (const c10::Error&) {
                // Expected for invalid configurations
            }
        }
        
        // Try with a vector of output sizes
        if (offset + sizeof(int64_t) <= Size) {
            int64_t output_size_vec_raw;
            std::memcpy(&output_size_vec_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            output_size_vec_raw = (std::abs(output_size_vec_raw) % 100) + 1;
            
            std::vector<int64_t> output_size_vec = {output_size_vec_raw};
            
            torch::nn::AdaptiveAvgPool1d pool3(output_size_vec);
            
            try {
                torch::Tensor output3 = pool3(input);
                (void)output3.numel();
            } catch (const c10::Error&) {
                // Expected for invalid configurations
            }
        }
        
        // Try with 2D unbatched input
        if (input.dim() == 3 && input.size(0) == 1) {
            torch::Tensor input_2d = input.squeeze(0);  // (C, L)
            torch::nn::AdaptiveAvgPool1d pool_2d(output_size);
            
            try {
                torch::Tensor output_2d = pool_2d(input_2d);
                (void)output_2d.numel();
            } catch (const c10::Error&) {
                // Expected for invalid configurations
            }
        }
        
        // Test with output size equal to input size (identity-like operation)
        if (input.size(-1) > 0) {
            int64_t same_size = input.size(-1);
            torch::nn::AdaptiveAvgPool1d pool_same(same_size);
            
            try {
                torch::Tensor output_same = pool_same(input);
                (void)output_same.numel();
            } catch (const c10::Error&) {
                // Expected for invalid configurations
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