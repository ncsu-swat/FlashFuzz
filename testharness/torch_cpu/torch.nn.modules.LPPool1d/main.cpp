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
        
        // Ensure we have at least 3 more bytes for parameters
        if (offset + 3 > Size) {
            return 0;
        }
        
        // Extract parameters for LPPool1d
        uint8_t norm_type_byte = Data[offset++];
        uint8_t kernel_size_byte = Data[offset++];
        uint8_t stride_byte = Data[offset++];
        
        // Normalize parameters
        double norm_type = static_cast<double>(norm_type_byte % 10) + 1.0; // Norm type between 1 and 10
        int64_t kernel_size = static_cast<int64_t>(kernel_size_byte % 10) + 1; // Kernel size between 1 and 10
        int64_t stride = static_cast<int64_t>(stride_byte % 10) + 1; // Stride between 1 and 10
        
        // Create LPPool1d module
        torch::nn::LPPool1d lppool(
            torch::nn::LPPool1dOptions(norm_type, kernel_size)
                .stride(stride)
        );
        
        // Apply LPPool1d to the input tensor
        torch::Tensor output = lppool->forward(input);
        
        // Ensure the output is valid by performing a simple operation
        auto sum = output.sum();
        
        // Check if the sum is finite
        if (!std::isfinite(sum.item<float>())) {
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
