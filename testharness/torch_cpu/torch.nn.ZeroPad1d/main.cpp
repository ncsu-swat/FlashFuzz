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
        
        // Need at least a few bytes to create a tensor and padding values
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 2 bytes left for padding values
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Extract padding values from the input data
        int64_t padding_left = static_cast<int64_t>(Data[offset++]);
        int64_t padding_right = static_cast<int64_t>(Data[offset++]);
        
        // Create padding tuple
        std::vector<int64_t> padding;
        
        // Decide between single value or pair based on remaining byte if available
        if (offset < Size) {
            uint8_t padding_type = Data[offset++];
            if (padding_type % 2 == 0) {
                // Use a single padding value
                padding = {static_cast<int64_t>(padding_left)};
            } else {
                // Use a pair of padding values
                padding = {static_cast<int64_t>(padding_left), static_cast<int64_t>(padding_right)};
            }
        } else {
            // Default to pair if no more data
            padding = {static_cast<int64_t>(padding_left), static_cast<int64_t>(padding_right)};
        }
        
        // Create ZeroPad1d module with padding parameter
        torch::nn::ZeroPad1d zero_pad(torch::nn::ZeroPad1dOptions(padding));
        
        // Apply padding
        torch::Tensor output_tensor = zero_pad->forward(input_tensor);
        
        // Force computation to ensure any errors are triggered
        output_tensor.sum().item<float>();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
