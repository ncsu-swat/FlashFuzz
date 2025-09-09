#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor shape and data type
        auto shape = parse_tensor_shape(Data, Size, offset);
        if (shape.empty()) return 0;
        
        auto dtype = parse_dtype(Data, Size, offset);
        
        // Create input tensor
        torch::Tensor input = create_tensor(shape, dtype, Data, Size, offset);
        if (!input.defined()) return 0;
        
        // Parse dimension parameter
        int64_t dim = parse_int64(Data, Size, offset);
        
        // Clamp dimension to valid range
        int64_t ndim = input.dim();
        if (ndim == 0) return 0; // scalar tensors don't support cummin
        
        dim = ((dim % ndim) + ndim) % ndim; // Handle negative dimensions
        
        // Test cummin with dimension parameter
        auto result1 = torch::cummin(input, dim);
        auto values1 = std::get<0>(result1);
        auto indices1 = std::get<1>(result1);
        
        // Verify output shapes match input
        if (!values1.sizes().equals(input.sizes()) || 
            !indices1.sizes().equals(input.sizes())) {
            return -1;
        }
        
        // Verify indices are valid
        if (indices1.min().item<int64_t>() < 0 || 
            indices1.max().item<int64_t>() >= input.size(dim)) {
            return -1;
        }
        
        // Test with out parameter variant if we have enough data
        if (offset < Size) {
            torch::Tensor values_out = torch::empty_like(input);
            torch::Tensor indices_out = torch::empty_like(input, torch::kLong);
            
            torch::cummin_out(values_out, indices_out, input, dim);
            
            // Verify out variant produces same results
            if (!torch::allclose(values1, values_out, 1e-5, 1e-8, /*equal_nan=*/true) ||
                !torch::equal(indices1, indices_out)) {
                return -1;
            }
        }
        
        // Test edge cases with different tensor properties
        if (input.numel() > 0) {
            // Test with contiguous tensor
            auto contiguous_input = input.contiguous();
            auto result_contiguous = torch::cummin(contiguous_input, dim);
            
            // Test with non-contiguous tensor (if possible)
            if (input.dim() > 1) {
                auto transposed = input.transpose(0, std::min(1L, input.dim() - 1));
                if (!transposed.is_contiguous()) {
                    auto result_non_contiguous = torch::cummin(transposed, 
                        dim == 0 ? 1 : (dim == 1 ? 0 : dim));
                }
            }
        }
        
        // Test with different dimension values if tensor has multiple dimensions
        if (input.dim() > 1) {
            for (int64_t test_dim = 0; test_dim < input.dim(); ++test_dim) {
                auto result_dim = torch::cummin(input, test_dim);
                auto values_dim = std::get<0>(result_dim);
                auto indices_dim = std::get<1>(result_dim);
                
                // Basic sanity checks
                if (!values_dim.sizes().equals(input.sizes()) ||
                    !indices_dim.sizes().equals(input.sizes())) {
                    return -1;
                }
            }
        }
        
        // Test with negative dimension indexing
        int64_t neg_dim = dim - input.dim();
        auto result_neg = torch::cummin(input, neg_dim);
        auto values_neg = std::get<0>(result_neg);
        auto indices_neg = std::get<1>(result_neg);
        
        // Should produce same results as positive indexing
        if (!torch::allclose(values1, values_neg, 1e-5, 1e-8, /*equal_nan=*/true) ||
            !torch::equal(indices1, indices_neg)) {
            return -1;
        }
        
        // Test memory layout preservation
        if (input.is_cuda() == values1.is_cuda() && 
            input.device() == values1.device() &&
            input.dtype() == values1.dtype()) {
            // Device and dtype should be preserved for values
        } else {
            return -1;
        }
        
        // Indices should always be long type
        if (indices1.dtype() != torch::kLong) {
            return -1;
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}