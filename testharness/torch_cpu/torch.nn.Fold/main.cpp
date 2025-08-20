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
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Fold module
        // We need at least 8 bytes for the parameters
        if (offset + 8 > Size) {
            return 0;
        }
        
        // Extract output_size
        int64_t output_size_h = static_cast<int64_t>(Data[offset++]) % 64;
        int64_t output_size_w = static_cast<int64_t>(Data[offset++]) % 64;
        std::vector<int64_t> output_size = {output_size_h, output_size_w};
        
        // Extract kernel_size
        int64_t kernel_size_h = static_cast<int64_t>(Data[offset++]) % 16 + 1;
        int64_t kernel_size_w = static_cast<int64_t>(Data[offset++]) % 16 + 1;
        std::vector<int64_t> kernel_size = {kernel_size_h, kernel_size_w};
        
        // Extract dilation
        int64_t dilation_h = static_cast<int64_t>(Data[offset++]) % 8 + 1;
        int64_t dilation_w = static_cast<int64_t>(Data[offset++]) % 8 + 1;
        std::vector<int64_t> dilation = {dilation_h, dilation_w};
        
        // Extract padding
        int64_t padding_h = static_cast<int64_t>(Data[offset++]) % 8;
        int64_t padding_w = static_cast<int64_t>(Data[offset++]) % 8;
        std::vector<int64_t> padding = {padding_h, padding_w};
        
        // Extract stride
        int64_t stride_h = 1;
        int64_t stride_w = 1;
        if (offset + 2 <= Size) {
            stride_h = static_cast<int64_t>(Data[offset++]) % 8 + 1;
            stride_w = static_cast<int64_t>(Data[offset++]) % 8 + 1;
        }
        std::vector<int64_t> stride = {stride_h, stride_w};
        
        // Create Fold module
        torch::nn::Fold fold_module = torch::nn::Fold(
            torch::nn::FoldOptions(output_size, kernel_size)
                .dilation(dilation)
                .padding(padding)
                .stride(stride)
        );
        
        // Apply the fold operation
        torch::Tensor output = fold_module->forward(input);
        
        // Try to access the output tensor to ensure computation is performed
        if (output.defined()) {
            auto sizes = output.sizes();
            auto numel = output.numel();
            auto dtype = output.dtype();
            
            // Force evaluation of the tensor
            if (numel > 0) {
                auto sum = output.sum().item<float>();
                volatile float dummy = sum;
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