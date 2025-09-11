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
        
        // Ensure we have at least 3 bytes left for unfold parameters
        if (Size - offset < 3) {
            return 0;
        }
        
        // Parse unfold parameters
        int64_t kernel_size = static_cast<int64_t>(Data[offset++]) % 5 + 1;
        int64_t dilation = static_cast<int64_t>(Data[offset++]) % 3 + 1;
        int64_t padding = static_cast<int64_t>(Data[offset++]) % 3;
        int64_t stride = static_cast<int64_t>(Data[offset++]) % 3 + 1;
        
        // Create Unfold module with various parameters
        torch::nn::Unfold unfold = torch::nn::Unfold(
            torch::nn::UnfoldOptions(kernel_size)
                .dilation(dilation)
                .padding(padding)
                .stride(stride)
        );
        
        // Apply the unfold operation
        torch::Tensor output = unfold->forward(input);
        
        // Try with different kernel_size configurations
        if (offset < Size) {
            int64_t kernel_h = static_cast<int64_t>(Data[offset++]) % 4 + 1;
            int64_t kernel_w = static_cast<int64_t>(Data[offset++]) % 4 + 1;
            
            torch::nn::Unfold unfold2 = torch::nn::Unfold(
                torch::nn::UnfoldOptions({kernel_h, kernel_w})
                    .dilation(dilation)
                    .padding(padding)
                    .stride(stride)
            );
            
            torch::Tensor output2 = unfold2->forward(input);
        }
        
        // Try with different dilation configurations
        if (offset + 1 < Size) {
            int64_t dilation_h = static_cast<int64_t>(Data[offset++]) % 3 + 1;
            int64_t dilation_w = static_cast<int64_t>(Data[offset++]) % 3 + 1;
            
            torch::nn::Unfold unfold3 = torch::nn::Unfold(
                torch::nn::UnfoldOptions(kernel_size)
                    .dilation({dilation_h, dilation_w})
                    .padding(padding)
                    .stride(stride)
            );
            
            torch::Tensor output3 = unfold3->forward(input);
        }
        
        // Try with different padding configurations
        if (offset + 1 < Size) {
            int64_t padding_h = static_cast<int64_t>(Data[offset++]) % 3;
            int64_t padding_w = static_cast<int64_t>(Data[offset++]) % 3;
            
            torch::nn::Unfold unfold4 = torch::nn::Unfold(
                torch::nn::UnfoldOptions(kernel_size)
                    .dilation(dilation)
                    .padding({padding_h, padding_w})
                    .stride(stride)
            );
            
            torch::Tensor output4 = unfold4->forward(input);
        }
        
        // Try with different stride configurations
        if (offset + 1 < Size) {
            int64_t stride_h = static_cast<int64_t>(Data[offset++]) % 3 + 1;
            int64_t stride_w = static_cast<int64_t>(Data[offset++]) % 3 + 1;
            
            torch::nn::Unfold unfold5 = torch::nn::Unfold(
                torch::nn::UnfoldOptions(kernel_size)
                    .dilation(dilation)
                    .padding(padding)
                    .stride({stride_h, stride_w})
            );
            
            torch::Tensor output5 = unfold5->forward(input);
        }
        
        // Try with negative values for parameters (edge cases)
        if (offset < Size) {
            int64_t neg_param = -static_cast<int64_t>(Data[offset++] % 3 + 1);
            
            try {
                torch::nn::Unfold unfold_neg = torch::nn::Unfold(
                    torch::nn::UnfoldOptions(neg_param)
                        .dilation(dilation)
                        .padding(padding)
                        .stride(stride)
                );
                
                torch::Tensor output_neg = unfold_neg->forward(input);
            } catch (const std::exception &) {
                // Expected exception for negative kernel size
            }
            
            try {
                torch::nn::Unfold unfold_neg2 = torch::nn::Unfold(
                    torch::nn::UnfoldOptions(kernel_size)
                        .dilation(neg_param)
                        .padding(padding)
                        .stride(stride)
                );
                
                torch::Tensor output_neg2 = unfold_neg2->forward(input);
            } catch (const std::exception &) {
                // Expected exception for negative dilation
            }
            
            try {
                torch::nn::Unfold unfold_neg3 = torch::nn::Unfold(
                    torch::nn::UnfoldOptions(kernel_size)
                        .dilation(dilation)
                        .padding(neg_param)
                        .stride(stride)
                );
                
                torch::Tensor output_neg3 = unfold_neg3->forward(input);
            } catch (const std::exception &) {
                // Expected exception for negative padding
            }
            
            try {
                torch::nn::Unfold unfold_neg4 = torch::nn::Unfold(
                    torch::nn::UnfoldOptions(kernel_size)
                        .dilation(dilation)
                        .padding(padding)
                        .stride(neg_param)
                );
                
                torch::Tensor output_neg4 = unfold_neg4->forward(input);
            } catch (const std::exception &) {
                // Expected exception for negative stride
            }
        }
        
        // Try with zero values for parameters (edge cases)
        try {
            torch::nn::Unfold unfold_zero = torch::nn::Unfold(
                torch::nn::UnfoldOptions(0)
                    .dilation(dilation)
                    .padding(padding)
                    .stride(stride)
            );
            
            torch::Tensor output_zero = unfold_zero->forward(input);
        } catch (const std::exception &) {
            // Expected exception for zero kernel size
        }
        
        try {
            torch::nn::Unfold unfold_zero2 = torch::nn::Unfold(
                torch::nn::UnfoldOptions(kernel_size)
                    .dilation(0)
                    .padding(padding)
                    .stride(stride)
            );
            
            torch::Tensor output_zero2 = unfold_zero2->forward(input);
        } catch (const std::exception &) {
            // Expected exception for zero dilation
        }
        
        try {
            torch::nn::Unfold unfold_zero3 = torch::nn::Unfold(
                torch::nn::UnfoldOptions(kernel_size)
                    .dilation(dilation)
                    .padding(padding)
                    .stride(0)
            );
            
            torch::Tensor output_zero3 = unfold_zero3->forward(input);
        } catch (const std::exception &) {
            // Expected exception for zero stride
        }
        
        // Try with very large values
        if (offset < Size) {
            int64_t large_value = static_cast<int64_t>(Data[offset++]) + 1000;
            
            try {
                torch::nn::Unfold unfold_large = torch::nn::Unfold(
                    torch::nn::UnfoldOptions(large_value)
                        .dilation(dilation)
                        .padding(padding)
                        .stride(stride)
                );
                
                torch::Tensor output_large = unfold_large->forward(input);
            } catch (const std::exception &) {
                // May throw for very large kernel size
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
