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
        
        // Need at least a few bytes to create a meaningful test
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has at least 2D for AvgPool2d
        if (input.dim() < 2) {
            // Reshape to at least 2D if needed
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                // Scalar tensor, reshape to 1x1
                new_shape = {1, 1};
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to Nx1
                new_shape = {input.size(0), 1};
            }
            input = input.reshape(new_shape);
        }
        
        // Extract parameters for AvgPool2d from the remaining data
        if (offset + 8 <= Size) {
            // Extract kernel size
            int64_t kernel_h = 1, kernel_w = 1;
            std::memcpy(&kernel_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure kernel size is positive and reasonable
            kernel_h = std::abs(kernel_h) % 7 + 1;
            
            // Use same kernel size for both dimensions or extract another
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&kernel_w, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                kernel_w = std::abs(kernel_w) % 7 + 1;
            } else {
                kernel_w = kernel_h;
            }
            
            // Extract stride
            int64_t stride_h = 1, stride_w = 1;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&stride_h, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                stride_h = std::abs(stride_h) % 5 + 1;
                
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&stride_w, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    stride_w = std::abs(stride_w) % 5 + 1;
                } else {
                    stride_w = stride_h;
                }
            }
            
            // Extract padding
            int64_t padding_h = 0, padding_w = 0;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&padding_h, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                padding_h = std::abs(padding_h) % 3;
                
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&padding_w, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    padding_w = std::abs(padding_w) % 3;
                } else {
                    padding_w = padding_h;
                }
            }
            
            // Extract ceil_mode and count_include_pad
            bool ceil_mode = false;
            bool count_include_pad = true;
            if (offset < Size) {
                ceil_mode = Data[offset++] & 1;
                if (offset < Size) {
                    count_include_pad = Data[offset++] & 1;
                }
            }
            
            // Create AvgPool2d module with various configurations
            torch::nn::AvgPool2d avg_pool = nullptr;
            
            // Try different configurations based on the data
            if (offset % 3 == 0) {
                // Single kernel size
                avg_pool = torch::nn::AvgPool2d(
                    torch::nn::AvgPool2dOptions(kernel_h)
                        .stride(stride_h)
                        .padding(padding_h)
                        .ceil_mode(ceil_mode)
                        .count_include_pad(count_include_pad));
            } else if (offset % 3 == 1) {
                // Different kernel sizes for height and width
                avg_pool = torch::nn::AvgPool2d(
                    torch::nn::AvgPool2dOptions({kernel_h, kernel_w})
                        .stride({stride_h, stride_w})
                        .padding({padding_h, padding_w})
                        .ceil_mode(ceil_mode)
                        .count_include_pad(count_include_pad));
            } else {
                // Minimal configuration
                avg_pool = torch::nn::AvgPool2d(
                    torch::nn::AvgPool2dOptions(kernel_h));
            }
            
            // Apply the AvgPool2d operation
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
