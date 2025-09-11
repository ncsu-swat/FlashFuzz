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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for ChannelShuffle
        // Need at least 2 more bytes for groups and dim parameters
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Get groups parameter (positive integer)
        int64_t groups = static_cast<int64_t>(Data[offset++]) % 16 + 1;
        
        // Get dimension parameter (typically 1 for 3D input, 0 for 2D input)
        int64_t dim = static_cast<int64_t>(Data[offset++]) % 4;
        
        // Apply ChannelShuffle operation using functional API
        torch::Tensor output;
        
        // Try different ways to call the operation
        if (offset < Size) {
            uint8_t call_type = Data[offset++] % 2;
            
            switch (call_type) {
                case 0:
                    // Standard functional call
                    output = torch::channel_shuffle(input, groups);
                    break;
                    
                case 1:
                    // Validate input dimensions and channels before calling
                    if (input.dim() > 0 && input.size(0) > 0) {
                        // Ensure we have a valid channel dimension
                        int64_t channels = 0;
                        if (input.dim() > 1) {
                            channels = input.size(1);
                        }
                        
                        // Only proceed if channels is divisible by groups
                        if (channels > 0 && channels % groups == 0) {
                            output = torch::channel_shuffle(input, groups);
                        } else {
                            // Adjust groups to be compatible
                            int64_t adjusted_groups = channels > 0 ? channels : 1;
                            output = torch::channel_shuffle(input, adjusted_groups);
                        }
                    } else {
                        output = torch::channel_shuffle(input, groups);
                    }
                    break;
            }
        } else {
            // Default to standard functional call
            output = torch::channel_shuffle(input, groups);
        }
        
        // Basic validation - output should have same shape as input
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
