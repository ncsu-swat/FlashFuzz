#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract norm_type (p) parameter first
        double norm_type = 2.0;
        if (offset < Size) {
            uint8_t p_byte = Data[offset++];
            // Use values between 1 and 6 for norm_type
            norm_type = 1.0 + (p_byte % 6);
        }
        
        // Extract kernel_size
        int64_t kernel_size = 2;
        if (offset < Size) {
            kernel_size = (Data[offset++] % 5) + 1;  // 1-5
        }
        
        // Extract stride (0 means use kernel_size as stride)
        int64_t stride = 0;
        if (offset < Size) {
            uint8_t stride_byte = Data[offset++];
            if (stride_byte & 0x80) {
                stride = (stride_byte % 5) + 1;  // 1-5
            }
            // else stride = 0, which means use kernel_size
        }
        
        // Extract ceil_mode
        bool ceil_mode = false;
        if (offset < Size) {
            ceil_mode = Data[offset++] & 0x1;
        }
        
        // Create input tensor from remaining data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if needed (LPPool2d requires floating point)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure input has 4 dimensions (N, C, H, W) for LPPool2d
        while (input.dim() < 4) {
            input = input.unsqueeze(0);
        }
        
        // Ensure spatial dimensions are large enough for kernel
        int64_t H = input.size(2);
        int64_t W = input.size(3);
        if (H < kernel_size || W < kernel_size) {
            // Expand spatial dimensions if needed
            int64_t new_H = std::max(H, kernel_size + 1);
            int64_t new_W = std::max(W, kernel_size + 1);
            input = torch::nn::functional::pad(input, 
                torch::nn::functional::PadFuncOptions({0, new_W - W, 0, new_H - H}));
        }
        
        // Create LPPool2d with options - basic configuration
        {
            auto options = torch::nn::LPPool2dOptions(norm_type, kernel_size);
            if (stride > 0) {
                options.stride(stride);
            }
            options.ceil_mode(ceil_mode);
            
            torch::nn::LPPool2d lppool(options);
            
            try {
                torch::Tensor output = lppool->forward(input);
                // Basic sanity check
                (void)output.numel();
            } catch (const c10::Error&) {
                // Shape mismatch or other PyTorch errors - expected
            }
        }
        
        // Try with 2D kernel_size and stride configurations
        if (offset + 4 <= Size) {
            int64_t kernel_h = (Data[offset++] % 4) + 1;  // 1-4
            int64_t kernel_w = (Data[offset++] % 4) + 1;  // 1-4
            int64_t stride_h = (Data[offset++] % 4) + 1;  // 1-4
            int64_t stride_w = (Data[offset++] % 4) + 1;  // 1-4
            
            // Ensure input is large enough
            H = input.size(2);
            W = input.size(3);
            if (H >= kernel_h && W >= kernel_w) {
                auto options2 = torch::nn::LPPool2dOptions(norm_type, {kernel_h, kernel_w})
                    .stride({stride_h, stride_w})
                    .ceil_mode(ceil_mode);
                
                torch::nn::LPPool2d lppool2(options2);
                
                try {
                    torch::Tensor output2 = lppool2->forward(input);
                    (void)output2.numel();
                } catch (const c10::Error&) {
                    // Expected for some configurations
                }
            }
        }
        
        // Try with different norm_type values
        if (offset < Size) {
            double alt_norm_type = (Data[offset++] % 4) + 1;  // 1, 2, 3, or 4
            
            auto options3 = torch::nn::LPPool2dOptions(alt_norm_type, kernel_size);
            torch::nn::LPPool2d lppool3(options3);
            
            try {
                torch::Tensor output3 = lppool3->forward(input);
                (void)output3.numel();
            } catch (const c10::Error&) {
                // Expected for some configurations
            }
        }
        
        // Test with 3D input (unbatched)
        if (input.dim() == 4 && input.size(0) == 1) {
            torch::Tensor input3d = input.squeeze(0);
            
            auto options4 = torch::nn::LPPool2dOptions(norm_type, kernel_size);
            torch::nn::LPPool2d lppool4(options4);
            
            try {
                torch::Tensor output4 = lppool4->forward(input3d);
                (void)output4.numel();
            } catch (const c10::Error&) {
                // Expected - some configurations may not work with 3D
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}