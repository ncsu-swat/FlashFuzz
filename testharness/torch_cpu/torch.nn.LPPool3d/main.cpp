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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 3D tensor for LPPool3d
        if (input.dim() < 3) {
            // Expand dimensions to make it at least 3D
            while (input.dim() < 3) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract parameters for LPPool3d from the remaining data
        if (offset + 8 > Size) {
            return 0;
        }
        
        // Extract norm_type (1 or 2 are common values)
        int64_t norm_type_raw;
        std::memcpy(&norm_type_raw, Data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        
        // Ensure norm_type is positive (typically 1 or 2, but allow other values for fuzzing)
        double norm_type = std::abs(static_cast<double>(norm_type_raw % 10)) + 0.1;
        
        // Extract kernel size
        uint8_t kernel_size_byte = (offset < Size) ? Data[offset++] : 1;
        int kernel_size = (kernel_size_byte % 5) + 1; // Kernel size between 1 and 5
        
        // Extract stride (default to kernel_size if not enough data)
        int stride = kernel_size;
        if (offset < Size) {
            stride = (Data[offset++] % 5) + 1; // Stride between 1 and 5
        }
        
        // Extract padding (default to 0 if not enough data)
        int padding = 0;
        if (offset < Size) {
            padding = Data[offset++] % 3; // Padding between 0 and 2
        }
        
        // Extract ceil_mode (default to false if not enough data)
        bool ceil_mode = false;
        if (offset < Size) {
            ceil_mode = (Data[offset++] % 2) == 1;
        }
        
        // Create LPPool3d module using options
        torch::nn::LPPool3dOptions options(norm_type, kernel_size);
        options.stride(stride);
        options.ceil_mode(ceil_mode);
        torch::nn::LPPool3d lp_pool(options);
        
        // Apply LPPool3d to the input tensor
        torch::Tensor output = lp_pool->forward(input);
        
        // Try with different kernel size configurations
        if (offset + 2 < Size) {
            // Try with tuple kernel_size
            int k1 = (Data[offset] % 5) + 1;
            int k2 = (Data[offset+1] % 5) + 1;
            int k3 = (Data[offset+2] % 5) + 1;
            offset += 3;
            
            torch::nn::LPPool3dOptions options_tuple(norm_type, std::vector<int64_t>{k1, k2, k3});
            options_tuple.stride(stride);
            options_tuple.ceil_mode(ceil_mode);
            torch::nn::LPPool3d lp_pool_tuple(options_tuple);
            torch::Tensor output_tuple = lp_pool_tuple->forward(input);
        }
        
        // Try with different stride configurations
        if (offset + 2 < Size) {
            // Try with tuple stride
            int s1 = (Data[offset] % 5) + 1;
            int s2 = (Data[offset+1] % 5) + 1;
            int s3 = (Data[offset+2] % 5) + 1;
            offset += 3;
            
            torch::nn::LPPool3dOptions options_stride(norm_type, kernel_size);
            options_stride.stride(std::vector<int64_t>{s1, s2, s3});
            options_stride.ceil_mode(ceil_mode);
            torch::nn::LPPool3d lp_pool_stride(options_stride);
            torch::Tensor output_stride = lp_pool_stride->forward(input);
        }
        
        // Try with both tuple kernel_size and tuple stride
        if (offset + 5 < Size) {
            int k1 = (Data[offset] % 5) + 1;
            int k2 = (Data[offset+1] % 5) + 1;
            int k3 = (Data[offset+2] % 5) + 1;
            int s1 = (Data[offset+3] % 5) + 1;
            int s2 = (Data[offset+4] % 5) + 1;
            int s3 = (Data[offset+5] % 5) + 1;
            
            torch::nn::LPPool3dOptions options_both(norm_type, std::vector<int64_t>{k1, k2, k3});
            options_both.stride(std::vector<int64_t>{s1, s2, s3});
            options_both.ceil_mode(ceil_mode);
            torch::nn::LPPool3d lp_pool_both(options_both);
            torch::Tensor output_both = lp_pool_both->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
