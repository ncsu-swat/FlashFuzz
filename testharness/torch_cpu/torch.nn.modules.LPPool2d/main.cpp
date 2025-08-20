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
        
        // Ensure input has at least 3 dimensions (N, C, H, W) for LPPool2d
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        
        // Extract parameters for LPPool2d from the remaining data
        if (offset + 4 > Size) {
            return 0;
        }
        
        // Extract norm_type (p) parameter
        double norm_type = 2.0;
        if (offset < Size) {
            uint8_t p_byte = Data[offset++];
            // Use values between 1 and 6 for norm_type
            norm_type = 1.0 + (p_byte % 6);
        }
        
        // Extract kernel_size
        int64_t kernel_size = 2;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure kernel_size is positive and reasonable
            kernel_size = std::abs(kernel_size) % 5 + 1;
        }
        
        // Create LPPool2d module with simple constructor
        torch::nn::LPPool2d lppool(norm_type, kernel_size);
        
        // Apply LPPool2d to the input tensor
        torch::Tensor output = lppool->forward(input);
        
        // Try different configurations
        if (offset < Size) {
            // Try with different kernel_size and stride configurations
            int64_t kernel_h = 2, kernel_w = 3;
            int64_t stride_h = 2, stride_w = 2;
            
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&kernel_h, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                kernel_h = std::abs(kernel_h) % 5 + 1;
            }
            
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&kernel_w, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                kernel_w = std::abs(kernel_w) % 5 + 1;
            }
            
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&stride_h, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                stride_h = std::abs(stride_h) % 5 + 1;
            }
            
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&stride_w, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                stride_w = std::abs(stride_w) % 5 + 1;
            }
            
            // Extract ceil_mode flag
            bool ceil_mode = false;
            if (offset < Size) {
                ceil_mode = Data[offset++] & 0x1;
            }
            
            // Create LPPool2d with options
            auto options = torch::nn::LPPool2dOptions({static_cast<double>(kernel_h), static_cast<double>(kernel_w)})
                .stride({static_cast<double>(stride_h), static_cast<double>(stride_w)})
                .ceil_mode(ceil_mode);
            
            torch::nn::LPPool2d lppool2(options);
            
            // Apply the second LPPool2d
            torch::Tensor output2 = lppool2->forward(input);
        }
        
        // Try with a different norm_type
        if (offset < Size) {
            double alt_norm_type = 1.0;
            if (norm_type == 1.0) {
                alt_norm_type = 3.0;
            }
            
            torch::nn::LPPool2d lppool3(alt_norm_type, kernel_size);
            torch::Tensor output3 = lppool3->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}