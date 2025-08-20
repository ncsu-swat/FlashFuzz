#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor for MaxUnpool2d
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create indices tensor for MaxUnpool2d
        torch::Tensor indices;
        if (offset < Size) {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create indices with same shape as input
            indices = torch::zeros_like(input, torch::kLong);
        }
        
        // Extract parameters for MaxUnpool2d
        int64_t kernel_size_h = 2;
        int64_t kernel_size_w = 2;
        int64_t stride_h = 2;
        int64_t stride_w = 2;
        int64_t padding_h = 0;
        int64_t padding_w = 0;
        
        // If we have more data, use it for parameters
        if (offset + 6 <= Size) {
            kernel_size_h = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            kernel_size_w = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            stride_h = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            stride_w = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            padding_h = static_cast<int64_t>(Data[offset++]) % 3;
            padding_w = static_cast<int64_t>(Data[offset++]) % 3;
        }
        
        // Create output_size for MaxUnpool2d
        std::vector<int64_t> output_size;
        if (input.dim() >= 2) {
            // Try to create a valid output size based on input dimensions
            for (int i = 0; i < input.dim(); i++) {
                if (i >= input.dim() - 2) {
                    // For the last two dimensions, calculate a reasonable output size
                    int64_t dim_value = input.size(i) * stride_h;
                    output_size.push_back(dim_value);
                } else {
                    // For other dimensions, keep the same size
                    output_size.push_back(input.size(i));
                }
            }
        }
        
        // Create MaxUnpool2d module
        torch::nn::MaxUnpool2d unpool = torch::nn::MaxUnpool2d(
            torch::nn::MaxUnpool2dOptions({kernel_size_h, kernel_size_w})
                .stride({stride_h, stride_w})
                .padding({padding_h, padding_w})
        );
        
        // Apply MaxUnpool2d operation
        torch::Tensor output;
        
        // Try different ways to call MaxUnpool2d
        if (offset % 3 == 0 && !output_size.empty()) {
            // Call with output_size
            output = unpool->forward(input, indices, output_size);
        } else {
            // Call without output_size
            output = unpool->forward(input, indices);
        }
        
        // Try to access output properties to ensure computation completed
        auto output_sizes = output.sizes();
        auto output_dtype = output.dtype();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}