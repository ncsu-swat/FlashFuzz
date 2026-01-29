#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cstdint>
#include <limits>

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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor (must have at least 1 dimension for select_scatter)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // select_scatter requires input to have at least 1 dimension
        if (input.dim() == 0) {
            input = input.unsqueeze(0);
        }
        
        // Get dim parameter - constrain to valid range
        int64_t dim = 0;
        if (offset < Size) {
            dim = static_cast<int64_t>(Data[offset]) % input.dim();
            offset++;
        }
        
        // Get index parameter - constrain to valid range for the selected dimension
        int64_t index = 0;
        if (offset < Size && input.size(dim) > 0) {
            index = static_cast<int64_t>(Data[offset]) % input.size(dim);
            offset++;
        }
        
        // Create src tensor with correct shape
        // src should have the same shape as input.select(dim, index)
        // which means it has one fewer dimension than input
        torch::Tensor src;
        if (offset < Size) {
            src = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            src = torch::ones({1});
        }
        
        // Reshape src to match the expected shape for select_scatter
        // The expected shape is input's shape with dimension `dim` removed
        std::vector<int64_t> expected_shape;
        for (int64_t i = 0; i < input.dim(); i++) {
            if (i != dim) {
                expected_shape.push_back(input.size(i));
            }
        }
        
        // If expected_shape is empty (input was 1D), src should be a scalar
        if (expected_shape.empty()) {
            src = src.flatten()[0];  // Get scalar
        } else {
            // Try to reshape src to expected shape, or create a new tensor
            int64_t total_elements = 1;
            for (auto s : expected_shape) {
                total_elements *= s;
            }
            if (total_elements > 0) {
                // Create src with the correct shape
                src = torch::zeros(expected_shape, input.options());
            }
        }
        
        // Apply select_scatter operation with valid parameters
        try {
            torch::Tensor result = torch::select_scatter(input, src, dim, index);
        } catch (const c10::Error& e) {
            // Expected for some input combinations
        }
        
        // Try with negative dimension (valid in PyTorch)
        try {
            int64_t neg_dim = -(input.dim() - dim);
            if (neg_dim != 0) {
                torch::Tensor result = torch::select_scatter(input, src, neg_dim, index);
            }
        } catch (const c10::Error& e) {
            // Expected for invalid inputs
        }
        
        // Try with negative index (valid in PyTorch for indexing from end)
        try {
            if (input.size(dim) > 0) {
                int64_t neg_index = -(input.size(dim) - index);
                if (neg_index != 0) {
                    torch::Tensor result = torch::select_scatter(input, src, dim, neg_index);
                }
            }
        } catch (const c10::Error& e) {
            // Expected for invalid inputs
        }
        
        // Test with different dtypes
        try {
            torch::Tensor float_input = input.to(torch::kFloat32);
            torch::Tensor float_src = src.to(torch::kFloat32);
            torch::Tensor result = torch::select_scatter(float_input, float_src, dim, index);
        } catch (const c10::Error& e) {
            // Expected for invalid inputs
        }
        
        // Test with fuzzer-provided raw dim and index (for edge case exploration)
        if (offset + 2 <= Size) {
            int8_t raw_dim = static_cast<int8_t>(Data[offset]);
            int8_t raw_index = static_cast<int8_t>(Data[offset + 1]);
            offset += 2;
            
            try {
                // Recreate src for potentially different dim
                int64_t test_dim = raw_dim;
                if (test_dim >= 0 && test_dim < input.dim()) {
                    std::vector<int64_t> test_shape;
                    for (int64_t i = 0; i < input.dim(); i++) {
                        if (i != test_dim) {
                            test_shape.push_back(input.size(i));
                        }
                    }
                    torch::Tensor test_src = test_shape.empty() ? 
                        torch::tensor(1.0) : 
                        torch::zeros(test_shape, input.options());
                    torch::Tensor result = torch::select_scatter(input, test_src, test_dim, raw_index);
                }
            } catch (const c10::Error& e) {
                // Expected for invalid inputs
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