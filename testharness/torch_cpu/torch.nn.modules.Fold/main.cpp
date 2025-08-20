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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Fold module
        // We need at least 8 bytes for the parameters
        if (offset + 8 >= Size) {
            return 0;
        }
        
        // Extract output_size parameters
        int64_t output_height = static_cast<int64_t>(Data[offset++]) % 64;
        int64_t output_width = static_cast<int64_t>(Data[offset++]) % 64;
        
        // Extract kernel_size parameters
        int64_t kernel_height = 1 + (static_cast<int64_t>(Data[offset++]) % 8);
        int64_t kernel_width = 1 + (static_cast<int64_t>(Data[offset++]) % 8);
        
        // Extract dilation parameters
        int64_t dilation_height = 1 + (static_cast<int64_t>(Data[offset++]) % 4);
        int64_t dilation_width = 1 + (static_cast<int64_t>(Data[offset++]) % 4);
        
        // Extract padding parameters
        int64_t padding_height = static_cast<int64_t>(Data[offset++]) % 4;
        int64_t padding_width = static_cast<int64_t>(Data[offset++]) % 4;
        
        // Extract stride parameters
        int64_t stride_height = 1 + (static_cast<int64_t>(Data[offset++]) % 4);
        int64_t stride_width = 1 + (static_cast<int64_t>(Data[offset++]) % 4);
        
        // Create parameter vectors
        std::vector<int64_t> output_size = {output_height, output_width};
        std::vector<int64_t> kernel_size = {kernel_height, kernel_width};
        std::vector<int64_t> dilation = {dilation_height, dilation_width};
        std::vector<int64_t> padding = {padding_height, padding_width};
        std::vector<int64_t> stride = {stride_height, stride_width};
        
        // Create Fold module
        torch::nn::Fold fold_module(
            torch::nn::FoldOptions(output_size, kernel_size)
                .dilation(dilation)
                .padding(padding)
                .stride(stride)
        );
        
        // Apply the fold operation
        torch::Tensor output = fold_module->forward(input);
        
        // Try with different parameter combinations
        if (offset + 5 < Size) {
            // Create a different set of parameters
            int64_t alt_output_size = 1 + (static_cast<int64_t>(Data[offset++]) % 32);
            int64_t alt_kernel_size = 1 + (static_cast<int64_t>(Data[offset++]) % 8);
            int64_t alt_dilation = 1 + (static_cast<int64_t>(Data[offset++]) % 4);
            int64_t alt_padding = static_cast<int64_t>(Data[offset++]) % 4;
            int64_t alt_stride = 1 + (static_cast<int64_t>(Data[offset++]) % 4);
            
            // Create a fold module with scalar parameters
            torch::nn::Fold alt_fold_module(
                torch::nn::FoldOptions({alt_output_size, alt_output_size}, alt_kernel_size)
                    .dilation(alt_dilation)
                    .padding(alt_padding)
                    .stride(alt_stride)
            );
            
            // Apply the fold operation
            torch::Tensor alt_output = alt_fold_module->forward(input);
        }
        
        // Try with edge case parameters
        if (offset + 1 < Size) {
            uint8_t edge_case = Data[offset++];
            
            if (edge_case % 4 == 0) {
                // Try with minimum valid parameters
                torch::nn::Fold min_fold_module(
                    torch::nn::FoldOptions({1, 1}, 1)
                        .dilation(1)
                        .padding(0)
                        .stride(1)
                );
                torch::Tensor min_output = min_fold_module->forward(input);
            }
            else if (edge_case % 4 == 1) {
                // Try with large output size
                torch::nn::Fold large_fold_module(
                    torch::nn::FoldOptions({100, 100}, 3)
                        .dilation(1)
                        .padding(0)
                        .stride(1)
                );
                torch::Tensor large_output = large_fold_module->forward(input);
            }
            else if (edge_case % 4 == 2) {
                // Try with large kernel size
                torch::nn::Fold large_kernel_module(
                    torch::nn::FoldOptions({20, 20}, 10)
                        .dilation(1)
                        .padding(0)
                        .stride(1)
                );
                torch::Tensor large_kernel_output = large_kernel_module->forward(input);
            }
            else {
                // Try with large dilation
                torch::nn::Fold large_dilation_module(
                    torch::nn::FoldOptions({20, 20}, 3)
                        .dilation(5)
                        .padding(0)
                        .stride(1)
                );
                torch::Tensor large_dilation_output = large_dilation_module->forward(input);
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