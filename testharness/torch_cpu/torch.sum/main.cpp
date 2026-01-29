#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

// --- Fuzzer Entry Point ---
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
        
        // Extract parameters for sum operation if we have more data
        int64_t dim = 0;
        bool keepdim = false;
        
        if (offset < Size) {
            // Extract dimension parameter - normalize to valid range
            uint8_t dim_byte = Data[offset++];
            if (input.dim() > 0) {
                dim = static_cast<int64_t>(dim_byte % input.dim());
                // Allow negative dimensions too (50% chance)
                if (offset < Size && (Data[offset] & 0x1)) {
                    dim = -(input.dim() - dim);
                }
            }
            
            // Extract keepdim parameter if we have more data
            if (offset < Size) {
                keepdim = Data[offset++] & 0x1;
            }
        }
        
        // Variant 1: Sum over all dimensions (reduce to scalar)
        try {
            torch::Tensor result1 = torch::sum(input);
            (void)result1;
        } catch (const c10::Error &e) {
            // Expected for some edge cases
        }
        
        // Variant 2: Sum over specific dimension using IntArrayRef
        if (input.dim() > 0) {
            try {
                torch::Tensor result2 = torch::sum(input, {dim}, keepdim);
                (void)result2;
            } catch (const c10::Error &e) {
                // Expected for invalid dimensions
            }
        }
        
        // Variant 3: Sum with dtype conversion
        try {
            torch::Tensor result3 = torch::sum(input, torch::kFloat);
            (void)result3;
        } catch (const c10::Error &e) {
            // Expected for incompatible dtypes
        }
        
        try {
            torch::Tensor result4 = torch::sum(input, torch::kDouble);
            (void)result4;
        } catch (const c10::Error &e) {
            // Expected for incompatible dtypes
        }
        
        // Variant 4: Sum over multiple dimensions if tensor has enough dims
        if (input.dim() >= 2) {
            try {
                std::vector<int64_t> dims = {0, 1};
                torch::Tensor result5 = torch::sum(input, dims, keepdim);
                (void)result5;
            } catch (const c10::Error &e) {
                // Expected for shape issues
            }
        }
        
        // Variant 5: Sum with out tensor
        try {
            torch::Tensor out = torch::empty({}, input.options());
            torch::sum_out(out, input);
            (void)out;
        } catch (const c10::Error &e) {
            // Expected for incompatible output tensors
        }
        
        // Variant 6: Sum over dimension with out tensor
        if (input.dim() > 0) {
            try {
                // Calculate expected output shape
                auto input_sizes = input.sizes().vec();
                std::vector<int64_t> out_sizes;
                for (int64_t i = 0; i < input.dim(); i++) {
                    if (i == (dim < 0 ? input.dim() + dim : dim)) {
                        if (keepdim) {
                            out_sizes.push_back(1);
                        }
                    } else {
                        out_sizes.push_back(input_sizes[i]);
                    }
                }
                torch::Tensor out = torch::empty(out_sizes, input.options());
                torch::sum_out(out, input, {dim}, keepdim);
                (void)out;
            } catch (const c10::Error &e) {
                // Expected for shape mismatches
            }
        }
        
        // Variant 7: Sum with complex dtype if input is real
        if (!input.is_complex()) {
            try {
                torch::Tensor result7 = torch::sum(input, torch::kComplexFloat);
                (void)result7;
            } catch (const c10::Error &e) {
                // May not support complex conversion
            }
        }
        
        // Variant 8: Test with different integer dtypes
        try {
            torch::Tensor result8 = torch::sum(input, torch::kLong);
            (void)result8;
        } catch (const c10::Error &e) {
            // Expected for some dtypes
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}