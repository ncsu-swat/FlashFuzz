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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 5D tensor (batch, channels, depth, height, width)
        if (input.dim() < 5) {
            input = input.reshape({1, 1, 1, 1, 1});
        }
        
        // Extract parameters for LPPool3d from the remaining data
        uint8_t norm_type_byte = (offset < Size) ? Data[offset++] : 2;
        double norm_type = static_cast<double>(norm_type_byte % 10) + 1.0; // Norm type between 1 and 10
        
        // Extract kernel size
        int64_t kernel_d = 2, kernel_h = 2, kernel_w = 2;
        if (offset + 3 <= Size) {
            kernel_d = static_cast<int64_t>(Data[offset++] % 5) + 1; // 1-5
            kernel_h = static_cast<int64_t>(Data[offset++] % 5) + 1; // 1-5
            kernel_w = static_cast<int64_t>(Data[offset++] % 5) + 1; // 1-5
        }
        
        // Extract stride
        int64_t stride_d = 1, stride_h = 1, stride_w = 1;
        if (offset + 3 <= Size) {
            stride_d = static_cast<int64_t>(Data[offset++] % 3) + 1; // 1-3
            stride_h = static_cast<int64_t>(Data[offset++] % 3) + 1; // 1-3
            stride_w = static_cast<int64_t>(Data[offset++] % 3) + 1; // 1-3
        }
        
        // Extract ceil_mode
        bool ceil_mode = false;
        if (offset < Size) {
            ceil_mode = Data[offset++] % 2 == 1;
        }
        
        // Create LPPool3d module with different configurations
        torch::nn::LPPool3d lppool_single = nullptr;
        torch::nn::LPPool3d lppool_triple = nullptr;
        
        // Configure with single kernel size and stride
        lppool_single = torch::nn::LPPool3d(
            torch::nn::LPPool3dOptions(norm_type, kernel_d)
                .stride(stride_d)
                .ceil_mode(ceil_mode)
        );
        
        // Configure with triple kernel size and stride
        lppool_triple = torch::nn::LPPool3d(
            torch::nn::LPPool3dOptions(norm_type, {kernel_d, kernel_h, kernel_w})
                .stride({stride_d, stride_h, stride_w})
                .ceil_mode(ceil_mode)
        );
        
        // Apply the LPPool3d operations
        torch::Tensor output_single = lppool_single->forward(input);
        torch::Tensor output_triple = lppool_triple->forward(input);
        
        // Try functional version as well
        torch::Tensor output_functional = torch::nn::functional::lp_pool3d(
            input, 
            norm_type, 
            torch::nn::functional::LPPool3dFuncOptions({kernel_d, kernel_h, kernel_w})
                .stride({stride_d, stride_h, stride_w})
                .ceil_mode(ceil_mode)
        );
        
        // Ensure results are valid by performing a simple operation
        auto sum_single = output_single.sum();
        auto sum_triple = output_triple.sum();
        auto sum_functional = output_functional.sum();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
