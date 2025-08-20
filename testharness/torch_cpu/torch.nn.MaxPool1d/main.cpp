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
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for MaxPool1d from the remaining data
        if (offset + 8 > Size) {
            return 0;
        }
        
        // Extract kernel_size
        int32_t kernel_size;
        std::memcpy(&kernel_size, Data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        kernel_size = std::abs(kernel_size) % 16 + 1; // Ensure positive and reasonable size
        
        // Extract stride (default to kernel_size if not specified)
        int32_t stride;
        std::memcpy(&stride, Data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        stride = std::abs(stride) % 16 + 1; // Ensure positive and reasonable stride
        
        // Extract padding (default to 0)
        int32_t padding = 0;
        if (offset + sizeof(int32_t) <= Size) {
            std::memcpy(&padding, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            padding = std::abs(padding) % 8; // Ensure reasonable padding
        }
        
        // Extract dilation (default to 1)
        int32_t dilation = 1;
        if (offset + sizeof(int32_t) <= Size) {
            std::memcpy(&dilation, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            dilation = std::abs(dilation) % 8 + 1; // Ensure positive and reasonable dilation
        }
        
        // Extract ceil_mode (default to false)
        bool ceil_mode = false;
        if (offset < Size) {
            ceil_mode = Data[offset++] & 0x1; // Use lowest bit to determine boolean
        }
        
        // Create MaxPool1d module
        torch::nn::MaxPool1d pool(torch::nn::MaxPool1dOptions(kernel_size)
                                 .stride(stride)
                                 .padding(padding)
                                 .dilation(dilation)
                                 .ceil_mode(ceil_mode));
        
        // Apply MaxPool1d to the input tensor
        torch::Tensor output = pool->forward(input);
        
        // Ensure the output is materialized to catch any deferred errors
        output.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}