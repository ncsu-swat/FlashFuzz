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
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 2D tensor for MaxPool2d
        if (input_tensor.dim() < 2) {
            // Reshape to at least 2D if needed
            std::vector<int64_t> new_shape;
            if (input_tensor.dim() == 0) {
                // Scalar tensor, reshape to 1x1
                new_shape = {1, 1};
            } else if (input_tensor.dim() == 1) {
                // 1D tensor, reshape to Nx1
                new_shape = {input_tensor.size(0), 1};
            }
            input_tensor = input_tensor.reshape(new_shape);
        }
        
        // Extract parameters for MaxPool2d from the remaining data
        if (offset + 8 > Size) {
            return 0;
        }
        
        // Parse kernel size
        int64_t kernel_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure kernel size is positive and reasonable
            kernel_size = std::abs(kernel_size) % 7 + 1;
        }
        
        // Parse stride
        int64_t stride = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure stride is positive and reasonable
            stride = std::abs(stride) % 5 + 1;
        }
        
        // Parse padding
        int64_t padding = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Allow padding to be any value including negative
            padding = padding % 5;
        }
        
        // Parse dilation
        int64_t dilation = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dilation, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure dilation is positive and reasonable
            dilation = std::abs(dilation) % 3 + 1;
        }
        
        // Parse ceil_mode
        bool ceil_mode = false;
        if (offset < Size) {
            ceil_mode = Data[offset++] & 0x1;
        }
        
        // Create MaxPool2d module
        torch::nn::MaxPool2d max_pool(
            torch::nn::MaxPool2dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .ceil_mode(ceil_mode));
        
        // Apply MaxPool2d to the input tensor
        torch::Tensor output = max_pool(input_tensor);
        
        // Ensure the output is valid
        if (output.numel() > 0) {
            // Access some elements to ensure computation happened
            auto sum = output.sum().item<float>();
            (void)sum;  // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
