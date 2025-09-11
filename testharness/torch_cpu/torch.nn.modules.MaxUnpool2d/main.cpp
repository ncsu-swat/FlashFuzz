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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create indices tensor (same shape as input)
        torch::Tensor indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure indices are integers
        indices = indices.to(torch::kInt64);
        
        // Extract kernel size parameters
        std::vector<int64_t> kernel_size;
        if (offset + 2 <= Size) {
            uint8_t kernel_h = Data[offset++] % 8 + 1; // 1-8
            uint8_t kernel_w = Data[offset++] % 8 + 1; // 1-8
            kernel_size = {static_cast<int64_t>(kernel_h), static_cast<int64_t>(kernel_w)};
        } else {
            kernel_size = {2, 2}; // Default
        }
        
        // Extract stride parameters
        std::vector<int64_t> stride;
        if (offset + 2 <= Size) {
            uint8_t stride_h = Data[offset++] % 4 + 1; // 1-4
            uint8_t stride_w = Data[offset++] % 4 + 1; // 1-4
            stride = {static_cast<int64_t>(stride_h), static_cast<int64_t>(stride_w)};
        } else {
            stride = kernel_size; // Default to same as kernel_size
        }
        
        // Extract padding parameters
        std::vector<int64_t> padding;
        if (offset + 2 <= Size) {
            uint8_t padding_h = Data[offset++] % 4; // 0-3
            uint8_t padding_w = Data[offset++] % 4; // 0-3
            padding = {static_cast<int64_t>(padding_h), static_cast<int64_t>(padding_w)};
        } else {
            padding = {0, 0}; // Default
        }
        
        // Extract output_size parameters (optional)
        std::vector<int64_t> output_size;
        if (offset + 1 <= Size) {
            uint8_t use_output_size = Data[offset++] % 2; // 0 or 1
            
            if (use_output_size && input.dim() >= 2) {
                if (offset + 2 <= Size) {
                    uint8_t out_h = Data[offset++] % 32 + 1; // 1-32
                    uint8_t out_w = Data[offset++] % 32 + 1; // 1-32
                    
                    // Create output_size with batch and channel dimensions from input
                    if (input.dim() == 4) {
                        output_size = {input.size(0), input.size(1), 
                                      static_cast<int64_t>(out_h), 
                                      static_cast<int64_t>(out_w)};
                    } else if (input.dim() == 3) {
                        output_size = {input.size(0), 
                                      static_cast<int64_t>(out_h), 
                                      static_cast<int64_t>(out_w)};
                    } else {
                        output_size = {static_cast<int64_t>(out_h), 
                                      static_cast<int64_t>(out_w)};
                    }
                }
            }
        }
        
        // Create MaxUnpool2d module
        torch::nn::MaxUnpool2d unpool = torch::nn::MaxUnpool2d(
            torch::nn::MaxUnpool2dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
        );
        
        // Apply the operation
        torch::Tensor output;
        if (!output_size.empty()) {
            output = unpool->forward(input, indices, output_size);
        } else {
            output = unpool->forward(input, indices);
        }
        
        // Try to access output properties to ensure computation completed
        auto sizes = output.sizes();
        auto dtype = output.dtype();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
