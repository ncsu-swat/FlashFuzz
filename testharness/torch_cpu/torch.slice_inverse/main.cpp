#include "fuzzer_utils.h"
#include <iostream>
#include <algorithm>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Create the source tensor (the "original" tensor shape we want to inverse into)
        torch::Tensor src = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (src.dim() == 0 || src.numel() == 0) {
            return 0;
        }
        
        // Extract dim parameter
        int64_t dim = 0;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t dim_byte;
            std::memcpy(&dim_byte, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            dim = dim_byte % src.dim();
            if (dim < 0) dim += src.dim();
        }
        
        int64_t dim_size = src.size(dim);
        if (dim_size == 0) {
            return 0;
        }
        
        // Extract start parameter
        int64_t start = 0;
        if (offset + sizeof(int16_t) <= Size) {
            int16_t start_raw;
            std::memcpy(&start_raw, Data + offset, sizeof(int16_t));
            offset += sizeof(int16_t);
            start = start_raw % (dim_size + 1);
            if (start < 0) start += dim_size;
        }
        
        // Extract end parameter  
        int64_t end = dim_size;
        if (offset + sizeof(int16_t) <= Size) {
            int16_t end_raw;
            std::memcpy(&end_raw, Data + offset, sizeof(int16_t));
            offset += sizeof(int16_t);
            end = end_raw % (dim_size + 1);
            if (end < 0) end += dim_size;
        }
        
        // Extract step parameter (must be positive non-zero)
        int64_t step = 1;
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t step_raw;
            std::memcpy(&step_raw, Data + offset, sizeof(uint8_t));
            offset += sizeof(uint8_t);
            step = (step_raw % 4) + 1; // step between 1-4
        }
        
        // Ensure start < end for positive step
        if (start > end) {
            std::swap(start, end);
        }
        
        // First perform a slice to get the shape of the sliced tensor
        torch::Tensor sliced;
        try {
            sliced = torch::slice(src, dim, start, end, step);
        } catch (...) {
            return 0;
        }
        
        if (sliced.numel() == 0) {
            return 0;
        }
        
        // Create input tensor with same shape as sliced tensor
        // This is the tensor we want to "inverse" back
        torch::Tensor input = torch::randn_like(sliced);
        
        // Apply slice_inverse: places input back into a src-shaped tensor
        // at the positions corresponding to slice(src, dim, start, end, step)
        torch::Tensor result;
        try {
            result = torch::slice_inverse(input, src, dim, start, end, step);
        } catch (const c10::Error &e) {
            // Expected for some invalid parameter combinations
            return 0;
        }
        
        // Verify result
        if (result.defined() && result.numel() > 0) {
            // Check result has same shape as src
            if (result.sizes() != src.sizes()) {
                std::cerr << "Shape mismatch in result!" << std::endl;
            }
            auto sum = result.sum().item<float>();
            (void)sum;
        }
        
        // Test with optional none values
        if (offset < Size && (Data[offset % Size] & 0x1)) {
            try {
                auto result2 = torch::slice_inverse(input, src, dim, c10::nullopt, end, step);
                if (result2.defined() && result2.numel() > 0) {
                    auto s = result2.sum().item<float>();
                    (void)s;
                }
            } catch (...) {
                // Expected for some combinations
            }
        }
        
        if (offset < Size && (Data[offset % Size] & 0x2)) {
            try {
                auto result3 = torch::slice_inverse(input, src, dim, start, c10::nullopt, step);
                if (result3.defined() && result3.numel() > 0) {
                    auto s = result3.sum().item<float>();
                    (void)s;
                }
            } catch (...) {
                // Expected for some combinations
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