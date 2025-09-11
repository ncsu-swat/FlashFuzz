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
        int64_t kernel_size = static_cast<int64_t>(Data[offset++]) % 10 + 1;
        int64_t dilation = static_cast<int64_t>(Data[offset++]) % 5 + 1;
        int64_t padding = static_cast<int64_t>(Data[offset++]) % 5;
        int64_t stride = static_cast<int64_t>(Data[offset++]) % 5 + 1;
        
        // Create the Unfold module
        torch::nn::Unfold unfold = torch::nn::Unfold(
            torch::nn::UnfoldOptions(kernel_size)
                .dilation(dilation)
                .padding(padding)
                .stride(stride)
        );
        
        // Apply the unfold operation
        torch::Tensor output = unfold->forward(input);
        
        // Test with different parameters if we have more data
        if (Size - offset >= 4) {
            kernel_size = static_cast<int64_t>(Data[offset++]) % 10 + 1;
            dilation = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            padding = static_cast<int64_t>(Data[offset++]) % 5;
            stride = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            
            // Create another Unfold module with different parameters
            torch::nn::Unfold unfold2 = torch::nn::Unfold(
                torch::nn::UnfoldOptions({kernel_size, kernel_size})
                    .dilation({dilation, dilation})
                    .padding({padding, padding})
                    .stride({stride, stride})
            );
            
            // Apply the second unfold operation
            torch::Tensor output2 = unfold2->forward(input);
        }
        
        // Test with tuple parameters if we have more data
        if (Size - offset >= 8) {
            int64_t kernel_h = static_cast<int64_t>(Data[offset++]) % 10 + 1;
            int64_t kernel_w = static_cast<int64_t>(Data[offset++]) % 10 + 1;
            int64_t dilation_h = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            int64_t dilation_w = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            int64_t padding_h = static_cast<int64_t>(Data[offset++]) % 5;
            int64_t padding_w = static_cast<int64_t>(Data[offset++]) % 5;
            int64_t stride_h = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            int64_t stride_w = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            
            // Create another Unfold module with tuple parameters
            torch::nn::Unfold unfold3 = torch::nn::Unfold(
                torch::nn::UnfoldOptions({kernel_h, kernel_w})
                    .dilation({dilation_h, dilation_w})
                    .padding({padding_h, padding_w})
                    .stride({stride_h, stride_w})
            );
            
            // Apply the third unfold operation
            torch::Tensor output3 = unfold3->forward(input);
        }
        
        // Test with negative parameters to check error handling
        if (Size - offset >= 4) {
            int64_t neg_kernel = -static_cast<int64_t>(Data[offset++] % 5 + 1);
            int64_t neg_dilation = -static_cast<int64_t>(Data[offset++] % 5 + 1);
            int64_t neg_padding = -static_cast<int64_t>(Data[offset++] % 5 + 1);
            int64_t neg_stride = -static_cast<int64_t>(Data[offset++] % 5 + 1);
            
            try {
                torch::nn::Unfold unfold_neg = torch::nn::Unfold(
                    torch::nn::UnfoldOptions(neg_kernel)
                        .dilation(neg_dilation)
                        .padding(neg_padding)
                        .stride(neg_stride)
                );
                
                torch::Tensor output_neg = unfold_neg->forward(input);
            } catch (const c10::Error &e) {
                // Expected exception for negative parameters
            }
        }
        
        // Test with zero kernel size to check error handling
        if (Size - offset >= 1) {
            try {
                torch::nn::Unfold unfold_zero = torch::nn::Unfold(
                    torch::nn::UnfoldOptions(0)
                );
                
                torch::Tensor output_zero = unfold_zero->forward(input);
            } catch (const c10::Error &e) {
                // Expected exception for zero kernel size
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
