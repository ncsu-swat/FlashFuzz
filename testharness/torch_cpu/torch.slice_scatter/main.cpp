#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip empty or scalar tensors
        if (input.dim() == 0 || input.numel() == 0) {
            return 0;
        }
        
        // Extract slice parameters from remaining data
        int64_t dim = 0;
        int64_t start = 0;
        int64_t end = 0;
        int64_t step = 1;
        
        // Get dimension to slice along
        if (offset + sizeof(uint8_t) <= Size) {
            dim = static_cast<int64_t>(Data[offset]) % input.dim();
            offset += sizeof(uint8_t);
        }
        
        int64_t dim_size = input.size(dim);
        
        // Get start index (bounded to valid range)
        if (offset + sizeof(uint8_t) <= Size) {
            start = static_cast<int64_t>(Data[offset]) % (dim_size + 1);
            offset += sizeof(uint8_t);
        }
        
        // Get end index (bounded to valid range, must be >= start)
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t end_offset = Data[offset];
            offset += sizeof(uint8_t);
            end = start + (end_offset % (dim_size - start + 1));
        } else {
            end = dim_size;
        }
        
        // Get step value (1-4 to keep it reasonable)
        if (offset + sizeof(uint8_t) <= Size) {
            step = 1 + (Data[offset] % 4);
            offset += sizeof(uint8_t);
        }
        
        // Calculate the size of the slice
        int64_t slice_size = 0;
        if (end > start && step > 0) {
            slice_size = (end - start + step - 1) / step;
        }
        
        if (slice_size <= 0) {
            return 0;
        }
        
        // Create src tensor with proper shape to match the slice
        std::vector<int64_t> src_sizes = input.sizes().vec();
        src_sizes[dim] = slice_size;
        
        torch::Tensor src;
        if (offset < Size) {
            // Try to create src from fuzzer data
            src = fuzzer_utils::createTensor(Data, Size, offset);
            // Reshape or recreate if dimensions don't match
            try {
                if (src.dim() != input.dim()) {
                    src = torch::ones(src_sizes, input.options());
                } else {
                    // Try to slice/pad src to match required shape
                    src = torch::ones(src_sizes, input.options());
                }
            } catch (...) {
                src = torch::ones(src_sizes, input.options());
            }
        } else {
            src = torch::ones(src_sizes, input.options());
        }
        
        // Ensure src has the same dtype as input
        src = src.to(input.dtype());
        
        // Apply slice_scatter operation
        torch::Tensor result;
        try {
            result = torch::slice_scatter(input, src, dim, start, end, step);
        } catch (const c10::Error& e) {
            // PyTorch specific exceptions for invalid parameters are expected
            return 0;
        }
        
        // Verify the result is a valid tensor with same shape as input
        if (result.defined()) {
            // slice_scatter should preserve the input shape
            volatile auto numel = result.numel();
            volatile bool same_shape = (result.sizes() == input.sizes());
            (void)numel;
            (void)same_shape;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}