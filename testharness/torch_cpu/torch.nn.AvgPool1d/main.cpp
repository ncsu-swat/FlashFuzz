#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the input tensor has at least 3 dimensions (N, C, L)
        // If not, reshape it to a valid shape for AvgPool1d
        if (input.dim() < 3) {
            int64_t total_elements = input.numel();
            if (total_elements > 0) {
                // Reshape to a valid 3D shape (batch, channels, length)
                int64_t batch_size = 1;
                int64_t channels = 1;
                int64_t length = total_elements;
                
                // Distribute elements to make a valid shape
                if (total_elements >= 3) {
                    batch_size = 1;
                    channels = total_elements / 3;
                    length = 3;
                }
                
                input = input.reshape({batch_size, channels, length});
            } else {
                // Create a minimal valid tensor if empty
                input = torch::ones({1, 1, 1});
            }
        }
        
        // Extract parameters for AvgPool1d from the remaining data
        if (offset + 3 <= Size) {
            // Extract kernel size
            int64_t kernel_size = static_cast<int64_t>(Data[offset++]) % 10 + 1;
            
            // Extract stride (default is kernel_size)
            int64_t stride = (offset < Size) ? 
                static_cast<int64_t>(Data[offset++]) % 10 + 1 : kernel_size;
            
            // Extract padding
            int64_t padding = (offset < Size) ? 
                static_cast<int64_t>(Data[offset++]) % 5 : 0;
            
            // Extract ceil_mode
            bool ceil_mode = (offset < Size) ? 
                (Data[offset++] % 2 == 1) : false;
            
            // Extract count_include_pad
            bool count_include_pad = (offset < Size) ? 
                (Data[offset++] % 2 == 1) : true;
            
            // Create AvgPool1d module
            torch::nn::AvgPool1d avg_pool(torch::nn::AvgPool1dOptions(kernel_size)
                                          .stride(stride)
                                          .padding(padding)
                                          .ceil_mode(ceil_mode)
                                          .count_include_pad(count_include_pad));
            
            // Apply AvgPool1d
            torch::Tensor output = avg_pool->forward(input);
        } else {
            // Use default parameters if not enough data
            torch::nn::AvgPool1d avg_pool(torch::nn::AvgPool1dOptions(2));
            torch::Tensor output = avg_pool->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}