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
        size_t offset = 0;
        
        // Need at least 3 bytes for basic parameters
        if (Size < 3) {
            return 0;
        }
        
        // Parse parameters for ChannelShuffle
        uint8_t groups_byte = Data[offset++];
        int64_t groups = (groups_byte % 8) + 1; // Ensure at least 1 group, max 8
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // ChannelShuffle expects input of shape [N, C, H, W]
        // If tensor doesn't have 4 dimensions, reshape it
        if (input.dim() != 4) {
            std::vector<int64_t> new_shape;
            
            if (input.dim() == 0) {
                // Scalar tensor, reshape to [1, groups, 1, 1]
                new_shape = {1, groups, 1, 1};
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to [1, C, 1, 1]
                int64_t C = input.size(0);
                if (C == 0) C = groups;
                new_shape = {1, C, 1, 1};
            } else if (input.dim() == 2) {
                // 2D tensor, reshape to [N, C, 1, 1]
                int64_t N = input.size(0);
                int64_t C = input.size(1);
                if (N == 0) N = 1;
                if (C == 0) C = groups;
                new_shape = {N, C, 1, 1};
            } else if (input.dim() == 3) {
                // 3D tensor, reshape to [N, C, H, 1]
                int64_t N = input.size(0);
                int64_t C = input.size(1);
                int64_t H = input.size(2);
                if (N == 0) N = 1;
                if (C == 0) C = groups;
                if (H == 0) H = 1;
                new_shape = {N, C, H, 1};
            } else {
                // More than 4 dimensions, take first 4
                new_shape = {input.size(0), input.size(1), input.size(2), input.size(3)};
                for (auto& dim : new_shape) {
                    if (dim == 0) dim = 1;
                }
            }
            
            // Reshape tensor if possible
            try {
                int64_t total_elements = input.numel();
                int64_t new_elements = 1;
                for (auto s : new_shape) new_elements *= s;
                
                if (total_elements == new_elements && total_elements > 0) {
                    input = input.reshape(new_shape);
                } else {
                    input = torch::ones(new_shape, input.options());
                }
            } catch (const std::exception&) {
                // If reshape fails, create a new tensor
                input = torch::ones(new_shape, input.options());
            }
        }
        
        // Handle zero-sized dimensions
        if (input.numel() == 0) {
            input = torch::ones({1, groups, 1, 1}, input.options());
        }
        
        // Ensure channel dimension is divisible by groups
        int64_t channels = input.size(1);
        if (channels % groups != 0) {
            // Adjust groups to be a divisor of channels
            for (int64_t i = groups; i > 0; i--) {
                if (channels % i == 0) {
                    groups = i;
                    break;
                }
            }
            
            // If no divisor found, set groups to 1
            if (channels % groups != 0) {
                groups = 1;
            }
        }
        
        // Apply ChannelShuffle to input tensor using functional API
        torch::Tensor output = torch::channel_shuffle(input, groups);
        
        // Verify output shape matches input shape (channel shuffle should preserve shape)
        if (output.sizes() != input.sizes()) {
            throw std::runtime_error("Output shape doesn't match input shape");
        }
        
        // Additional coverage: test with different tensor types
        try {
            auto float_input = input.to(torch::kFloat);
            auto float_output = torch::channel_shuffle(float_input, groups);
            (void)float_output;
        } catch (const std::exception&) {
            // Some dtypes may not be supported, ignore silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}