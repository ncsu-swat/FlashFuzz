#include "fuzzer_utils.h"
#include <iostream>
#include <algorithm>
#include <vector>

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
        
        // Skip 0-dimensional tensors as amax needs at least 1 dimension
        if (input.dim() == 0) {
            return 0;
        }
        
        // Extract dim parameter
        int64_t dim = 0;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t raw_dim = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
            // Normalize to valid dimension range [-dim, dim-1]
            dim = raw_dim % input.dim();
        }
        
        // Extract keepdim parameter
        bool keepdim = false;
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x1);
            offset++;
        }
        
        // Variant 1: amax over single dimension
        try {
            torch::Tensor result1 = torch::amax(input, dim, keepdim);
            (void)result1;
        } catch (const c10::Error&) {
            // Expected for invalid dimension combinations
        }
        
        // Variant 2: amax over multiple dimensions (if tensor has at least 2 dimensions)
        if (input.dim() >= 2) {
            try {
                std::vector<int64_t> dims;
                
                // Determine how many dimensions to reduce over
                int num_dims = (offset < Size) ? (1 + (Data[offset++] % (input.dim() - 1))) : 1;
                
                for (int i = 0; i < num_dims && offset < Size; i++) {
                    int64_t d = static_cast<int64_t>(Data[offset++]) % input.dim();
                    dims.push_back(d);
                }
                
                // Remove duplicates
                std::sort(dims.begin(), dims.end());
                dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
                
                if (!dims.empty()) {
                    torch::Tensor result2 = torch::amax(input, dims, keepdim);
                    (void)result2;
                }
            } catch (const c10::Error&) {
                // Expected for invalid dimension combinations
            }
        }
        
        // Variant 3: amax_out variant
        try {
            // Normalize dimension for output shape calculation
            int64_t norm_dim = dim < 0 ? dim + input.dim() : dim;
            
            if (norm_dim >= 0 && norm_dim < input.dim()) {
                std::vector<int64_t> out_shape;
                
                if (keepdim) {
                    out_shape = input.sizes().vec();
                    out_shape[norm_dim] = 1;
                } else {
                    for (int64_t i = 0; i < input.dim(); i++) {
                        if (i != norm_dim) {
                            out_shape.push_back(input.size(i));
                        }
                    }
                }
                
                if (!out_shape.empty()) {
                    torch::Tensor output = torch::empty(out_shape, input.options());
                    torch::amax_out(output, input, dim, keepdim);
                } else {
                    // Result is a scalar (0-dim tensor)
                    torch::Tensor output = torch::empty({}, input.options());
                    torch::amax_out(output, input, dim, keepdim);
                }
            }
        } catch (const c10::Error&) {
            // Expected for shape mismatches or invalid configurations
        }
        
        // Variant 4: Test with IntArrayRef containing single dimension
        try {
            std::vector<int64_t> single_dim = {dim};
            torch::Tensor result4 = torch::amax(input, single_dim, keepdim);
            (void)result4;
        } catch (const c10::Error&) {
            // Expected for invalid dimension
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}