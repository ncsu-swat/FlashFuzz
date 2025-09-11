#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least 3 bytes for basic parameters
        if (Size < 3) {
            return 0;
        }
        
        // Parse parameters for ChannelShuffle
        uint8_t groups_byte = Data[offset++];
        int groups = (groups_byte % 8) + 1; // Ensure at least 1 group, max 8
        
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
            // Create a 4D tensor with appropriate shape
            std::vector<int64_t> new_shape;
            
            if (input.dim() == 0) {
                // Scalar tensor, reshape to [1, groups, 1, 1]
                new_shape = {1, groups, 1, 1};
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to [1, C, 1, 1]
                int64_t C = input.size(0);
                new_shape = {1, C, 1, 1};
            } else if (input.dim() == 2) {
                // 2D tensor, reshape to [N, C, 1, 1]
                int64_t N = input.size(0);
                int64_t C = input.size(1);
                new_shape = {N, C, 1, 1};
            } else if (input.dim() == 3) {
                // 3D tensor, reshape to [N, C, H, 1]
                int64_t N = input.size(0);
                int64_t C = input.size(1);
                int64_t H = input.size(2);
                new_shape = {N, C, H, 1};
            } else {
                // More than 4 dimensions, take first 4
                new_shape = {input.size(0), input.size(1), input.size(2), input.size(3)};
            }
            
            // Reshape tensor if possible
            try {
                input = input.reshape(new_shape);
            } catch (const std::exception&) {
                // If reshape fails, create a new tensor
                input = torch::ones(new_shape, input.options());
            }
        }
        
        // Ensure channel dimension is divisible by groups
        int64_t channels = input.size(1);
        if (channels % groups != 0) {
            // Adjust groups to be a divisor of channels
            for (int i = groups; i > 0; i--) {
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
        
        // Verify output shape matches input shape
        if (output.sizes() != input.sizes()) {
            throw std::runtime_error("Output shape doesn't match input shape");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
