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
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // AdaptiveMaxPool1d expects:
        // - 2D input: (C, L) - unbatched
        // - 3D input: (N, C, L) - batched
        if (input.dim() < 1) {
            // For 0-dim tensor, reshape to [1, 1, 1]
            input = input.reshape({1, 1, 1});
        } else if (input.dim() == 1) {
            // For 1-dim tensor, treat as (C, L) with L=numel, C=1
            int64_t len = input.size(0);
            if (len < 1) len = 1;
            input = input.reshape({1, len});
        } else if (input.dim() == 2) {
            // Already 2D (C, L), keep as is for unbatched input
        } else if (input.dim() > 3) {
            // Flatten extra dimensions into batch
            int64_t batch = 1;
            for (int i = 0; i < input.dim() - 2; i++) {
                batch *= input.size(i);
            }
            int64_t channels = input.size(-2);
            int64_t length = input.size(-1);
            if (channels < 1) channels = 1;
            if (length < 1) length = 1;
            input = input.reshape({batch, channels, length});
        }
        
        // Ensure input is float type (required for pooling)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat);
        }
        
        // Get output size from remaining data
        int64_t output_size = 1; // Default
        if (offset + sizeof(int32_t) <= Size) {
            int32_t raw_size;
            std::memcpy(&raw_size, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            
            // Ensure output_size is positive and reasonable
            output_size = (std::abs(raw_size) % 100) + 1;
        }
        
        // Create AdaptiveMaxPool1d module with options
        auto options = torch::nn::AdaptiveMaxPool1dOptions(output_size);
        torch::nn::AdaptiveMaxPool1d pool(options);
        
        // Apply the operation
        auto output = pool(input);
        
        // Force computation
        auto dummy = output.sum().item<float>();
        (void)dummy;
        
        // Test with indices using forward_with_indices
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                auto [out_with_indices, indices] = pool->forward_with_indices(input);
                auto dummy2 = out_with_indices.sum().item<float>();
                auto dummy3 = indices.sum().item<int64_t>();
                (void)dummy2;
                (void)dummy3;
            } catch (...) {
                // Inner catch - silent for expected failures
            }
        }
        
        // Try different data types
        if (offset < Size && input.dim() >= 2) {
            auto dtype_selector = Data[offset++] % 3;
            torch::ScalarType new_dtype;
            
            switch (dtype_selector) {
                case 0: new_dtype = torch::kFloat; break;
                case 1: new_dtype = torch::kDouble; break;
                case 2: new_dtype = torch::kFloat; break; // kHalf/kBFloat16 may not be supported on CPU
                default: new_dtype = torch::kFloat;
            }
            
            if (input.scalar_type() != new_dtype) {
                try {
                    auto converted_input = input.to(new_dtype);
                    auto converted_output = pool(converted_input);
                    auto dummy4 = converted_output.sum();
                    (void)dummy4;
                } catch (...) {
                    // Inner catch - silent for expected conversion/computation failures
                }
            }
        }
        
        // Try with different output sizes
        if (offset + sizeof(int32_t) <= Size) {
            int32_t raw_alt_size;
            std::memcpy(&raw_alt_size, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            
            // Clamp to positive reasonable value
            int64_t alt_output_size = (std::abs(raw_alt_size) % 100) + 1;
            
            auto alt_options = torch::nn::AdaptiveMaxPool1dOptions(alt_output_size);
            torch::nn::AdaptiveMaxPool1d alt_pool(alt_options);
            
            try {
                auto alt_output = alt_pool(input);
                auto dummy5 = alt_output.sum();
                (void)dummy5;
            } catch (...) {
                // Inner catch - silent for expected failures
            }
        }
        
        // Test with 2D unbatched input if we have a 3D input
        if (input.dim() == 3 && input.size(0) > 0) {
            try {
                auto unbatched = input[0]; // Get first batch element (C, L)
                auto unbatched_output = pool(unbatched);
                auto dummy6 = unbatched_output.sum();
                (void)dummy6;
            } catch (...) {
                // Inner catch - silent
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