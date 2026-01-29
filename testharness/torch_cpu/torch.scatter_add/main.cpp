#include "fuzzer_utils.h"
#include <iostream>
#include <algorithm>

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
        
        if (Size < 10) {
            return 0;
        }
        
        // Create the input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (input.numel() == 0 || input.dim() == 0) {
            return 0;
        }
        
        // Get a dimension to scatter along
        int64_t dim = 0;
        if (offset < Size) {
            dim = static_cast<int64_t>(Data[offset++]) % input.dim();
        }
        
        // Create index tensor with same number of dimensions as input
        // Index values must be in valid range [0, input.size(dim))
        torch::Tensor index;
        if (offset < Size) {
            torch::Tensor raw_index = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert to int64 and ensure same dimensions
            raw_index = raw_index.to(torch::kInt64);
            
            // Reshape to have same number of dimensions as input
            std::vector<int64_t> index_shape;
            for (int64_t i = 0; i < input.dim(); i++) {
                if (i < raw_index.dim()) {
                    index_shape.push_back(std::min(raw_index.size(i), input.size(i)));
                } else {
                    index_shape.push_back(1);
                }
            }
            
            // Create index with proper shape and valid values
            int64_t total_elements = 1;
            for (auto s : index_shape) total_elements *= s;
            if (total_elements == 0) total_elements = 1;
            
            index = torch::randint(0, std::max(input.size(dim), (int64_t)1), index_shape, torch::kInt64);
        } else {
            std::vector<int64_t> index_shape(input.dim(), 1);
            index = torch::zeros(index_shape, torch::kInt64);
        }
        
        // Create src tensor with same shape as index
        torch::Tensor src = torch::randn(index.sizes(), input.options());
        
        // Variant 1: In-place scatter_add_
        try {
            auto result1 = input.clone().scatter_add_(dim, index, src);
        } catch (...) {
            // Expected failures for invalid inputs
        }
        
        // Variant 2: Functional form torch::scatter_add
        try {
            auto result2 = torch::scatter_add(input, dim, index, src);
        } catch (...) {
            // Expected failures
        }
        
        // Variant 3: Try with negative dimension
        if (offset < Size && input.dim() > 0) {
            try {
                int64_t neg_dim = -(1 + (Data[offset++] % input.dim()));
                auto result3 = torch::scatter_add(input, neg_dim, index, src);
            } catch (...) {
                // Expected failures
            }
        }
        
        // Variant 4: scatter with reduce parameter (different API)
        if (offset < Size) {
            try {
                // torch::scatter with reduce mode
                auto result4 = torch::scatter(input, dim, index, src);
            } catch (...) {
                // Expected failures
            }
        }
        
        // Variant 5: Different dtypes
        if (offset < Size) {
            try {
                auto float_input = input.to(torch::kFloat32);
                auto float_src = src.to(torch::kFloat32);
                auto result5 = torch::scatter_add(float_input, dim, index, float_src);
            } catch (...) {
                // Expected failures
            }
            
            try {
                auto double_input = input.to(torch::kFloat64);
                auto double_src = src.to(torch::kFloat64);
                auto result6 = torch::scatter_add(double_input, dim, index, double_src);
            } catch (...) {
                // Expected failures
            }
        }
        
        // Variant 6: Multi-dimensional scatter
        if (offset < Size && input.dim() > 1) {
            try {
                int64_t other_dim = (dim + 1) % input.dim();
                auto result7 = torch::scatter_add(input, other_dim, index, src);
            } catch (...) {
                // Expected failures
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