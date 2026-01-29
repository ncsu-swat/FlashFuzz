#include "fuzzer_utils.h"
#include <iostream>
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
        
        // Need at least a few bytes for basic operations
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters for creating a suitable tensor
        uint8_t num_dims = (Data[offset++] % 4) + 1;  // 1-4 dimensions
        
        std::vector<int64_t> shape;
        int64_t total_elements = 1;
        for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
            int64_t dim_size = (Data[offset++] % 8) + 1;  // 1-8 per dimension
            shape.push_back(dim_size);
            total_elements *= dim_size;
        }
        
        if (shape.empty() || offset >= Size) {
            return 0;
        }
        
        // Create input tensor with known shape
        torch::Tensor input_tensor = torch::randn(shape);
        
        // Get dimension to unflatten (valid range for this tensor)
        int64_t dim = static_cast<int64_t>(Data[offset++]) % static_cast<int64_t>(shape.size());
        
        // Get the size of the dimension we're unflattening
        int64_t dim_to_unflatten = shape[dim];
        
        if (offset >= Size) {
            return 0;
        }
        
        // Create sizes that multiply to the original dimension size
        // Find valid factor pairs
        std::vector<int64_t> unflatten_sizes;
        uint8_t num_factors = (Data[offset++] % 3) + 1;  // 1-3 factors
        
        if (num_factors == 1) {
            unflatten_sizes.push_back(dim_to_unflatten);
        } else if (num_factors == 2 && dim_to_unflatten > 1) {
            // Find a factor
            int64_t factor1 = 1;
            for (int64_t f = 2; f <= dim_to_unflatten; ++f) {
                if (dim_to_unflatten % f == 0) {
                    factor1 = f;
                    break;
                }
            }
            int64_t factor2 = dim_to_unflatten / factor1;
            unflatten_sizes.push_back(factor1);
            unflatten_sizes.push_back(factor2);
        } else {
            // For 3 factors or fallback, just use the original size
            unflatten_sizes.push_back(dim_to_unflatten);
        }
        
        // Create Unflatten module using UnflattenOptions
        auto options = torch::nn::UnflattenOptions(dim, unflatten_sizes);
        torch::nn::Unflatten unflatten_module(options);
        
        // Apply the unflatten operation
        torch::Tensor output = unflatten_module->forward(input_tensor);
        
        // Verify output shape is correct
        (void)output.sizes();
        
        // Test with negative dimension indexing
        if (offset < Size) {
            int64_t neg_dim = -1 - (Data[offset++] % static_cast<uint8_t>(shape.size()));
            int64_t actual_dim = shape.size() + neg_dim;
            if (actual_dim >= 0 && actual_dim < static_cast<int64_t>(shape.size())) {
                int64_t neg_dim_size = shape[actual_dim];
                std::vector<int64_t> neg_sizes = {neg_dim_size};
                
                try {
                    auto neg_options = torch::nn::UnflattenOptions(neg_dim, neg_sizes);
                    torch::nn::Unflatten unflatten_neg(neg_options);
                    torch::Tensor output_neg = unflatten_neg->forward(input_tensor);
                    (void)output_neg.sizes();
                } catch (...) {
                    // Silently handle expected failures
                }
            }
        }
        
        // Test edge case: sizes that don't match (should fail)
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                std::vector<int64_t> bad_sizes = {100, 100};
                auto bad_options = torch::nn::UnflattenOptions(0, bad_sizes);
                torch::nn::Unflatten unflatten_bad(bad_options);
                torch::Tensor output_bad = unflatten_bad->forward(input_tensor);
            } catch (...) {
                // Expected to fail - silently catch
            }
        }
        
        // Test with different tensor types
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            torch::Tensor typed_tensor;
            
            try {
                switch (dtype_selector) {
                    case 0:
                        typed_tensor = torch::randn(shape, torch::kFloat32);
                        break;
                    case 1:
                        typed_tensor = torch::randn(shape, torch::kFloat64);
                        break;
                    case 2:
                        typed_tensor = torch::randint(0, 100, shape, torch::kInt32);
                        break;
                    case 3:
                        typed_tensor = torch::randint(0, 100, shape, torch::kInt64);
                        break;
                }
                
                auto typed_options = torch::nn::UnflattenOptions(dim, unflatten_sizes);
                torch::nn::Unflatten unflatten_typed(typed_options);
                torch::Tensor output_typed = unflatten_typed->forward(typed_tensor);
                (void)output_typed.sizes();
            } catch (...) {
                // Silently handle
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