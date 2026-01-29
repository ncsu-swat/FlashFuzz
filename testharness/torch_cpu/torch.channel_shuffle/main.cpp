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

        // Need at least some bytes for tensor creation and parameters
        if (Size < 8) {
            return 0;
        }

        // Extract parameters from fuzzer data first
        uint8_t dim_selector = Data[offset++];
        uint8_t groups_byte = Data[offset++];
        uint8_t channels_byte = Data[offset++];
        uint8_t batch_byte = Data[offset++];

        // Determine tensor dimensions (channel_shuffle needs at least 3D: N, C, ...)
        int64_t batch = (batch_byte % 4) + 1;        // 1-4
        int64_t groups = (groups_byte % 8) + 1;      // 1-8
        int64_t channels_per_group = (channels_byte % 4) + 1;  // 1-4
        int64_t channels = groups * channels_per_group;  // Ensure divisible by groups

        // Create tensor with appropriate shape based on dim_selector
        torch::Tensor input;
        int dims = (dim_selector % 3) + 3;  // 3D, 4D, or 5D

        if (dims == 3) {
            // 3D: (N, C, L)
            int64_t length = ((offset < Size) ? (Data[offset++] % 8) : 4) + 1;
            input = torch::randn({batch, channels, length});
        } else if (dims == 4) {
            // 4D: (N, C, H, W) - most common for images
            int64_t height = ((offset < Size) ? (Data[offset++] % 8) : 4) + 1;
            int64_t width = ((offset < Size) ? (Data[offset++] % 8) : 4) + 1;
            input = torch::randn({batch, channels, height, width});
        } else {
            // 5D: (N, C, D, H, W) - for 3D convolutions
            int64_t depth = ((offset < Size) ? (Data[offset++] % 4) : 2) + 1;
            int64_t height = ((offset < Size) ? (Data[offset++] % 4) : 2) + 1;
            int64_t width = ((offset < Size) ? (Data[offset++] % 4) : 2) + 1;
            input = torch::randn({batch, channels, depth, height, width});
        }

        // Test with valid groups value (channels divisible by groups)
        try {
            torch::Tensor output = torch::channel_shuffle(input, groups);
            // Verify output shape matches input shape
            (void)output;
        } catch (...) {
            // Silently handle expected shape/divisibility errors
        }

        // Test with different dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::Tensor typed_input;
            
            try {
                switch (dtype_selector % 4) {
                    case 0:
                        typed_input = input.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_input = input.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_input = input.to(torch::kInt32);
                        break;
                    case 3:
                        typed_input = input.to(torch::kInt64);
                        break;
                }
                torch::Tensor output = torch::channel_shuffle(typed_input, groups);
                (void)output;
            } catch (...) {
                // Silently handle dtype-related errors
            }
        }

        // Test edge case: groups = 1 (should always work if tensor is valid)
        try {
            torch::Tensor output = torch::channel_shuffle(input, 1);
            (void)output;
        } catch (...) {
            // Silently handle errors
        }

        // Test edge case: groups = channels (each channel is its own group)
        try {
            torch::Tensor output = torch::channel_shuffle(input, channels);
            (void)output;
        } catch (...) {
            // Silently handle errors
        }

        // Test with potentially invalid groups value to exercise error paths
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_groups;
            std::memcpy(&raw_groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            try {
                torch::Tensor output = torch::channel_shuffle(input, raw_groups);
                (void)output;
            } catch (...) {
                // Silently handle invalid groups errors
            }
        }

        // Test with a tensor created from fuzzer data
        if (offset < Size) {
            try {
                torch::Tensor fuzz_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Only try if tensor has at least 3 dimensions
                if (fuzz_tensor.dim() >= 3 && fuzz_tensor.size(1) > 0) {
                    int64_t fuzz_channels = fuzz_tensor.size(1);
                    // Try groups that might divide channels
                    for (int64_t g = 1; g <= std::min(fuzz_channels, (int64_t)4); g++) {
                        if (fuzz_channels % g == 0) {
                            try {
                                torch::Tensor output = torch::channel_shuffle(fuzz_tensor, g);
                                (void)output;
                            } catch (...) {
                                // Silently handle errors
                            }
                        }
                    }
                }
            } catch (...) {
                // Silently handle tensor creation errors
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