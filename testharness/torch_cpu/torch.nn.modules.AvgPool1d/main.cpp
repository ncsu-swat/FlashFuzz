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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, L) for AvgPool1d
        if (input.dim() < 1) {
            input = input.unsqueeze(0);
        }
        
        // Extract parameters for AvgPool1d from the remaining data
        uint8_t kernel_size = 1;
        uint8_t stride = 1;
        uint8_t padding = 0;
        bool ceil_mode = false;
        bool count_include_pad = true;
        
        if (offset + 1 < Size) {
            kernel_size = Data[offset++] % 8 + 1; // Kernel size between 1 and 8
        }
        
        if (offset + 1 < Size) {
            stride = Data[offset++] % 4 + 1; // Stride between 1 and 4
        }
        
        if (offset + 1 < Size) {
            padding = Data[offset++] % 4; // Padding between 0 and 3
        }
        
        if (offset + 1 < Size) {
            ceil_mode = Data[offset++] % 2 == 1; // Boolean for ceil_mode
        }
        
        if (offset + 1 < Size) {
            count_include_pad = Data[offset++] % 2 == 1; // Boolean for count_include_pad
        }
        
        // Create AvgPool1d module
        torch::nn::AvgPool1d avg_pool(torch::nn::AvgPool1dOptions(kernel_size)
                                      .stride(stride)
                                      .padding(padding)
                                      .ceil_mode(ceil_mode)
                                      .count_include_pad(count_include_pad));
        
        // Apply AvgPool1d to the input tensor
        torch::Tensor output = avg_pool->forward(input);
        
        // Test with different options
        if (offset + 1 < Size) {
            // Try with different kernel_size
            uint8_t alt_kernel_size = Data[offset++] % 8 + 1;
            torch::nn::AvgPool1d alt_pool1((torch::nn::AvgPool1dOptions(alt_kernel_size)));
            torch::Tensor alt_output1 = alt_pool1->forward(input);
        }
        
        if (offset + 1 < Size) {
            // Try with different stride
            uint8_t alt_stride = Data[offset++] % 4 + 1;
            torch::nn::AvgPool1d alt_pool2((torch::nn::AvgPool1dOptions(kernel_size).stride(alt_stride)));
            torch::Tensor alt_output2 = alt_pool2->forward(input);
        }
        
        if (offset + 1 < Size) {
            // Try with different padding
            uint8_t alt_padding = Data[offset++] % 4;
            torch::nn::AvgPool1d alt_pool3((torch::nn::AvgPool1dOptions(kernel_size).padding(alt_padding)));
            torch::Tensor alt_output3 = alt_pool3->forward(input);
        }
        
        if (offset + 1 < Size) {
            // Try with different ceil_mode
            bool alt_ceil_mode = Data[offset++] % 2 == 1;
            torch::nn::AvgPool1d alt_pool4((torch::nn::AvgPool1dOptions(kernel_size).ceil_mode(alt_ceil_mode)));
            torch::Tensor alt_output4 = alt_pool4->forward(input);
        }
        
        if (offset + 1 < Size) {
            // Try with different count_include_pad
            bool alt_count_include_pad = Data[offset++] % 2 == 1;
            torch::nn::AvgPool1d alt_pool5((torch::nn::AvgPool1dOptions(kernel_size).count_include_pad(alt_count_include_pad)));
            torch::Tensor alt_output5 = alt_pool5->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
