#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <vector>

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
        
        // Need at least a few bytes for basic operations
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters first to determine tensor shape
        uint8_t dim_byte = Data[offset++];
        uint8_t num_factors = (Data[offset++] % 3) + 2; // 2-4 factors
        
        // Parse factors for the unflattened dimension
        std::vector<int64_t> unflatten_sizes;
        int64_t product = 1;
        for (uint8_t i = 0; i < num_factors && offset < Size; ++i) {
            int64_t factor = (Data[offset++] % 4) + 1; // 1-4 for each factor
            unflatten_sizes.push_back(factor);
            product *= factor;
        }
        
        // If we couldn't parse enough factors, use defaults
        if (unflatten_sizes.size() < 2) {
            unflatten_sizes = {2, 3};
            product = 6;
        }
        
        // Determine tensor shape - we need one dimension to have size = product
        uint8_t num_dims = (Data[offset % Size] % 3) + 1; // 1-3 dimensions
        offset++;
        
        std::vector<int64_t> tensor_shape;
        int64_t target_dim = dim_byte % (num_dims + 1); // Which dimension to unflatten
        if (target_dim < 0) target_dim = 0;
        
        for (uint8_t i = 0; i < num_dims; ++i) {
            if (i == static_cast<uint8_t>(target_dim)) {
                tensor_shape.push_back(product); // This dimension will be unflattened
            } else {
                int64_t dim_size = (offset < Size) ? (Data[offset++] % 4) + 1 : 2;
                tensor_shape.push_back(dim_size);
            }
        }
        
        // Ensure we have at least one dimension
        if (tensor_shape.empty()) {
            tensor_shape.push_back(product);
            target_dim = 0;
        }
        
        // Create input tensor with the computed shape
        torch::Tensor input = torch::randn(tensor_shape);
        
        // Normalize dimension index
        int64_t dim = target_dim;
        if (dim >= static_cast<int64_t>(tensor_shape.size())) {
            dim = tensor_shape.size() - 1;
        }
        
        // Test with positive dimension
        {
            torch::nn::Unflatten unflatten(
                torch::nn::UnflattenOptions(dim, unflatten_sizes)
            );
            
            torch::Tensor output = unflatten->forward(input);
            
            // Verify output
            if (output.numel() > 0) {
                volatile float sum = output.sum().item<float>();
                (void)sum;
            }
        }
        
        // Test with negative dimension (equivalent)
        try {
            int64_t neg_dim = dim - static_cast<int64_t>(tensor_shape.size());
            torch::nn::Unflatten unflatten_neg(
                torch::nn::UnflattenOptions(neg_dim, unflatten_sizes)
            );
            
            torch::Tensor output_neg = unflatten_neg->forward(input);
            (void)output_neg;
        } catch (...) {
            // Silently ignore - negative dim edge cases
        }
        
        // Test with named dimension if possible (different constructor)
        // The named constructor requires namedshape_t: vector<pair<string, int64_t>>
        try {
            std::vector<std::pair<std::string, int64_t>> named_shape;
            for (size_t i = 0; i < unflatten_sizes.size(); ++i) {
                named_shape.push_back({"dim_" + std::to_string(i), unflatten_sizes[i]});
            }
            torch::nn::Unflatten unflatten_named(
                torch::nn::UnflattenOptions("dim_name", named_shape)
            );
            // This will fail without named tensors, which is expected
        } catch (...) {
            // Silently ignore - named tensors may not be supported
        }
        
        // Test edge case: single-element unflatten sizes with -1
        try {
            std::vector<int64_t> inferred_sizes = unflatten_sizes;
            if (!inferred_sizes.empty()) {
                inferred_sizes[0] = -1; // Let PyTorch infer this dimension
            }
            torch::nn::Unflatten unflatten_infer(
                torch::nn::UnflattenOptions(dim, inferred_sizes)
            );
            torch::Tensor output_infer = unflatten_infer->forward(input);
            (void)output_infer;
        } catch (...) {
            // Silently ignore inference failures
        }
        
        // Test with different tensor dtypes
        try {
            torch::Tensor input_double = input.to(torch::kDouble);
            torch::nn::Unflatten unflatten_double(
                torch::nn::UnflattenOptions(dim, unflatten_sizes)
            );
            torch::Tensor output_double = unflatten_double->forward(input_double);
            (void)output_double;
        } catch (...) {
            // Silently ignore dtype-related issues
        }
        
        // Test with integer tensor
        try {
            torch::Tensor input_int = torch::randint(0, 10, tensor_shape, torch::kInt32);
            torch::nn::Unflatten unflatten_int(
                torch::nn::UnflattenOptions(dim, unflatten_sizes)
            );
            torch::Tensor output_int = unflatten_int->forward(input_int);
            (void)output_int;
        } catch (...) {
            // Silently ignore dtype-related issues
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}