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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create indices tensor (same shape as input)
        torch::Tensor indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure indices are valid (non-negative integers)
        indices = indices.abs().to(torch::kInt64);
        
        // Extract parameters for MaxUnpool1d
        int64_t kernel_size = 0;
        int64_t stride = 0;
        int64_t padding = 0;
        
        // Parse kernel_size
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_size = std::abs(kernel_size) % 10 + 1; // Ensure positive and reasonable
        } else {
            kernel_size = 2; // Default value
        }
        
        // Parse stride
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride = std::abs(stride) % 10 + 1; // Ensure positive and reasonable
        } else {
            stride = kernel_size; // Default value
        }
        
        // Parse padding
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding = std::abs(padding) % 5; // Ensure non-negative and reasonable
        } else {
            padding = 0; // Default value
        }
        
        // Create output_size vector if there's enough data
        std::optional<std::vector<int64_t>> output_size = std::nullopt;
        if (input.dim() > 0 && offset + sizeof(int64_t) <= Size) {
            int64_t output_length;
            std::memcpy(&output_length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            output_length = std::abs(output_length) % 100 + 1; // Ensure positive and reasonable
            
            if (input.dim() == 3) {
                output_size = std::vector<int64_t>{input.size(0), input.size(1), output_length};
            } else if (input.dim() == 2) {
                output_size = std::vector<int64_t>{input.size(0), output_length};
            }
        }
        
        // Create MaxUnpool1d options
        torch::nn::MaxUnpool1dOptions options(kernel_size);
        options.stride(stride);
        options.padding(padding);
        
        // Create MaxUnpool1d module
        torch::nn::MaxUnpool1d unpool(options);
        
        // Apply MaxUnpool1d operation
        torch::Tensor output = unpool->forward(input, indices, output_size);
        
        // Access output properties to ensure computation is performed
        auto output_sizes = output.sizes();
        auto output_dtype = output.dtype();
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}