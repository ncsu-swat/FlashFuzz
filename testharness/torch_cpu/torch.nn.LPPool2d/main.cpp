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
        
        // Ensure we have at least 2 dimensions for 2D pooling
        if (input.dim() < 2) {
            input = input.unsqueeze(0).unsqueeze(0);
        }
        
        // Extract parameters for LPPool2d from the remaining data
        uint8_t norm_type_byte = 1;
        uint8_t kernel_size_byte = 2;
        uint8_t stride_byte = 1;
        uint8_t ceil_mode_byte = 0;
        
        if (offset < Size) norm_type_byte = Data[offset++];
        if (offset < Size) kernel_size_byte = Data[offset++];
        if (offset < Size) stride_byte = Data[offset++];
        if (offset < Size) ceil_mode_byte = Data[offset++];
        
        // Parse parameters
        double norm_type = static_cast<double>(norm_type_byte % 10) + 1.0; // Ensure norm_type >= 1
        int kernel_size = static_cast<int>(kernel_size_byte % 8) + 1; // Ensure kernel_size >= 1
        int stride = static_cast<int>(stride_byte % 8) + 1; // Ensure stride >= 1
        bool ceil_mode = (ceil_mode_byte % 2) == 1; // Convert to boolean
        
        // Create LPPool2d module
        torch::nn::LPPool2d lppool(
            torch::nn::LPPool2dOptions(norm_type, kernel_size).stride(stride).ceil_mode(ceil_mode)
        );
        
        // Apply LPPool2d to the input tensor
        torch::Tensor output = lppool->forward(input);
        
        // Try with different kernel size configurations
        if (Size > offset + 1) {
            int kernel_h = static_cast<int>(Data[offset++] % 5) + 1;
            int kernel_w = static_cast<int>(Data[offset++] % 5) + 1;
            
            // Create LPPool2d with different kernel sizes for height and width
            torch::nn::LPPool2d lppool2(
                torch::nn::LPPool2dOptions(norm_type, {kernel_h, kernel_w}).stride(stride).ceil_mode(ceil_mode)
            );
            
            // Apply the second LPPool2d
            torch::Tensor output2 = lppool2->forward(input);
        }
        
        // Try with different stride configurations
        if (Size > offset + 1) {
            int stride_h = static_cast<int>(Data[offset++] % 5) + 1;
            int stride_w = static_cast<int>(Data[offset++] % 5) + 1;
            
            // Create LPPool2d with different strides for height and width
            torch::nn::LPPool2d lppool3(
                torch::nn::LPPool2dOptions(norm_type, kernel_size).stride({stride_h, stride_w}).ceil_mode(ceil_mode)
            );
            
            // Apply the third LPPool2d
            torch::Tensor output3 = lppool3->forward(input);
        }
        
        // Try with extreme norm_type values
        if (Size > offset) {
            double extreme_norm = static_cast<double>(Data[offset++]) / 10.0;
            
            // Create LPPool2d with extreme norm type
            torch::nn::LPPool2d lppool4(
                torch::nn::LPPool2dOptions(extreme_norm, kernel_size).stride(stride).ceil_mode(ceil_mode)
            );
            
            // Apply the fourth LPPool2d (may throw exception for invalid norm_type)
            try {
                torch::Tensor output4 = lppool4->forward(input);
            } catch (const std::exception&) {
                // Expected for invalid norm_type
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}