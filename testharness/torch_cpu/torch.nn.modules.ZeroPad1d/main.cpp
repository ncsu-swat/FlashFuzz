#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor and padding values
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract padding values from the remaining data
        int64_t padding_left = 0;
        int64_t padding_right = 0;
        
        // Get padding values from the remaining data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_left, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_right, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create ZeroPad1d module - initialize with nullptr first
        torch::nn::ZeroPad1d pad = nullptr;
        
        // Try different padding configurations
        torch::Tensor output;
        
        // Case 1: Single integer padding (same on both sides)
        if (offset < Size) {
            int64_t single_padding = static_cast<int64_t>(Data[offset++]);
            pad = torch::nn::ZeroPad1d(single_padding);
            output = pad->forward(input);
        }
        
        // Case 2: Tuple padding (different left/right padding)
        if (offset + 1 < Size) {
            pad = torch::nn::ZeroPad1d(torch::nn::ZeroPad1dOptions({padding_left, padding_right}));
            output = pad->forward(input);
        }
        
        // Case 3: Try with a different input shape if possible
        if (offset < Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            pad = torch::nn::ZeroPad1d(torch::nn::ZeroPad1dOptions({padding_left, padding_right}));
            output = pad->forward(input2);
        }
        
        // Case 4: Try with a different padding configuration
        if (offset + 1 < Size) {
            int64_t alt_padding = static_cast<int64_t>(Data[offset]);
            pad = torch::nn::ZeroPad1d(alt_padding);
            output = pad->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}