#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to use for max operation (if needed)
        int64_t dim = 0;
        bool keepdim = false;
        
        // If we have more data, use it to determine dimension and keepdim
        if (offset + 2 <= Size) {
            dim = static_cast<int64_t>(Data[offset++]);
            keepdim = Data[offset++] & 0x1;
        }
        
        // Variant 1: max of all elements (returns single-element tensor)
        torch::Tensor result1 = torch::max(input);
        (void)result1;
        
        // Variant 2: max along dimension (returns tuple of values and indices)
        if (input.dim() > 0) {
            // Normalize dim to valid range
            int64_t valid_dim = dim % input.dim();
            if (valid_dim < 0) valid_dim += input.dim();
            
            auto result2 = torch::max(input, valid_dim, keepdim);
            torch::Tensor values = std::get<0>(result2);
            torch::Tensor indices = std::get<1>(result2);
            (void)values;
            (void)indices;
        }
        
        // Variant 3: element-wise maximum of two tensors (torch::maximum)
        if (offset + 4 <= Size) {
            try {
                torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Use torch::maximum for element-wise max of two tensors
                // This broadcasts if shapes are compatible
                torch::Tensor result3 = torch::maximum(input, other);
                (void)result3;
            } catch (const std::exception &) {
                // Shapes might not be broadcastable, which is expected
            }
        }
        
        // Variant 4: element-wise maximum with scalar using torch::clamp_min
        // (torch::max doesn't take a scalar directly, use clamp_min instead)
        if (offset < Size) {
            double scalar_value = static_cast<double>(Data[offset++]) / 10.0;
            torch::Tensor result4 = torch::clamp_min(input, scalar_value);
            (void)result4;
        }
        
        // Variant 5: max with output tensor (in-place style)
        if (input.dim() > 0 && input.numel() > 0) {
            int64_t valid_dim = 0;
            if (input.dim() > 1) {
                valid_dim = (offset < Size) ? (Data[offset++] % input.dim()) : 0;
            }
            
            torch::Tensor values_out = torch::empty({0}, input.options());
            torch::Tensor indices_out = torch::empty({0}, torch::kLong);
            
            try {
                auto result5 = torch::max_out(values_out, indices_out, input, valid_dim, keepdim);
                (void)result5;
            } catch (const std::exception &) {
                // May fail due to dtype or shape issues
            }
        }
        
        // Variant 6: amax - another way to compute max along dimensions
        if (input.dim() > 0) {
            int64_t valid_dim = dim % input.dim();
            if (valid_dim < 0) valid_dim += input.dim();
            
            torch::Tensor result6 = torch::amax(input, {valid_dim}, keepdim);
            (void)result6;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}