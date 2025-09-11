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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 1 byte left for kernel_size
        if (offset >= Size) {
            return 0;
        }
        
        // Parse MaxPool1d parameters
        int64_t kernel_size = static_cast<int64_t>(Data[offset++]) % 10 + 1; // 1-10
        
        // Optional parameters
        int64_t stride = (offset < Size) ? (static_cast<int64_t>(Data[offset++]) % 10 + 1) : kernel_size;
        int64_t padding = (offset < Size) ? (static_cast<int64_t>(Data[offset++]) % 5) : 0; // 0-4
        int64_t dilation = (offset < Size) ? (static_cast<int64_t>(Data[offset++]) % 5 + 1) : 1; // 1-5
        bool ceil_mode = (offset < Size) ? (Data[offset++] % 2 == 1) : false;
        
        // Create MaxPool1d module
        torch::nn::MaxPool1d pool(torch::nn::MaxPool1dOptions(kernel_size)
                                  .stride(stride)
                                  .padding(padding)
                                  .dilation(dilation)
                                  .ceil_mode(ceil_mode));
        
        // Apply MaxPool1d to the input tensor
        torch::Tensor output = pool->forward(input);
        
        // Ensure the output is valid
        if (output.numel() > 0) {
            auto max_val = torch::max(output).item<float>();
        }
        
        // Use functional API for max_pool1d_with_indices
        auto result_tuple = torch::nn::functional::max_pool1d_with_indices(
            input,
            torch::nn::functional::MaxPool1dFuncOptions(kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .ceil_mode(ceil_mode)
        );
        
        torch::Tensor output_with_indices = std::get<0>(result_tuple);
        torch::Tensor indices = std::get<1>(result_tuple);
        
        // Ensure the outputs are valid
        if (output_with_indices.numel() > 0 && indices.numel() > 0) {
            auto max_val = torch::max(output_with_indices).item<float>();
            auto max_idx = torch::max(indices).item<int64_t>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
