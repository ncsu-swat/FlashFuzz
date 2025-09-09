#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor dimensions and parameters
        if (Size < 32) {
            return 0;
        }

        // Parse input tensor dimensions and properties
        auto input_shape = parse_random_shape(Data, Size, offset, 1, 4, 1, 100);
        auto input_dtype = parse_random_dtype(Data, Size, offset);
        
        // Create input tensor
        auto input_tensor = create_random_tensor(input_shape, input_dtype, Data, Size, offset);
        
        // Parse indices tensor - should be 1D with integer type
        auto indices_size = parse_range(Data, Size, offset, 1, std::min(static_cast<size_t>(1000), input_tensor.numel()));
        std::vector<int64_t> indices_shape = {static_cast<int64_t>(indices_size)};
        
        // Create indices tensor with valid indices for the input tensor
        auto indices_tensor = torch::randint(0, input_tensor.numel(), indices_shape, torch::kLong);
        
        // Parse source tensor dimensions - should be compatible with indices
        std::vector<int64_t> source_shape;
        if (indices_size == 1) {
            // For single index, source can be scalar or have same shape as input except first dim
            bool use_scalar = parse_bool(Data, Size, offset);
            if (use_scalar) {
                source_shape = {};  // scalar
            } else {
                source_shape = input_shape;
                if (source_shape.size() > 0) {
                    source_shape[0] = 1;  // single element in first dimension
                }
            }
        } else {
            // For multiple indices, source should have shape [indices_size, ...] matching input shape
            source_shape = input_shape;
            if (source_shape.size() > 0) {
                source_shape[0] = indices_size;
            } else {
                source_shape = {indices_size};
            }
        }
        
        auto source_tensor = create_random_tensor(source_shape, input_dtype, Data, Size, offset);
        
        // Test basic put operation
        auto result1 = input_tensor.clone();
        result1.put_(indices_tensor, source_tensor);
        
        // Test with accumulate parameter
        bool accumulate = parse_bool(Data, Size, offset);
        auto result2 = input_tensor.clone();
        result2.put_(indices_tensor, source_tensor, accumulate);
        
        // Test non-inplace version
        auto result3 = torch::put(input_tensor, indices_tensor, source_tensor);
        auto result4 = torch::put(input_tensor, indices_tensor, source_tensor, accumulate);
        
        // Test edge cases with different tensor configurations
        
        // Test with empty indices
        if (parse_bool(Data, Size, offset)) {
            auto empty_indices = torch::empty({0}, torch::kLong);
            auto empty_source = torch::empty({0}, input_dtype);
            auto result_empty = input_tensor.clone();
            result_empty.put_(empty_indices, empty_source);
        }
        
        // Test with scalar source
        if (parse_bool(Data, Size, offset) && source_tensor.numel() > 0) {
            auto scalar_source = source_tensor.flatten()[0];
            auto result_scalar = input_tensor.clone();
            result_scalar.put_(indices_tensor, scalar_source);
        }
        
        // Test with different index ranges
        if (parse_bool(Data, Size, offset) && input_tensor.numel() > 1) {
            // Test with negative indices (should wrap around)
            auto neg_indices = torch::randint(-static_cast<int64_t>(input_tensor.numel()), 0, {indices_size}, torch::kLong);
            auto result_neg = input_tensor.clone();
            result_neg.put_(neg_indices, source_tensor);
        }
        
        // Test with repeated indices
        if (parse_bool(Data, Size, offset) && indices_size > 1) {
            auto repeated_indices = torch::zeros({indices_size}, torch::kLong);  // all zeros
            auto result_repeated = input_tensor.clone();
            result_repeated.put_(repeated_indices, source_tensor, true);  // accumulate=true for repeated indices
        }
        
        // Test with different device configurations if CUDA is available
        if (torch::cuda::is_available() && parse_bool(Data, Size, offset)) {
            try {
                auto cuda_input = input_tensor.to(torch::kCUDA);
                auto cuda_indices = indices_tensor.to(torch::kCUDA);
                auto cuda_source = source_tensor.to(torch::kCUDA);
                
                auto cuda_result = cuda_input.clone();
                cuda_result.put_(cuda_indices, cuda_source);
                
                // Test mixed device scenarios (should fail gracefully)
                try {
                    auto mixed_result = input_tensor.clone();  // CPU tensor
                    mixed_result.put_(cuda_indices, source_tensor);  // CUDA indices, CPU source
                } catch (...) {
                    // Expected to fail, ignore
                }
            } catch (...) {
                // CUDA operations might fail, ignore
            }
        }
        
        // Test with different data types for source (broadcasting scenarios)
        if (parse_bool(Data, Size, offset)) {
            auto different_dtype = (input_dtype == torch::kFloat) ? torch::kDouble : torch::kFloat;
            try {
                auto diff_source = source_tensor.to(different_dtype);
                auto result_diff = input_tensor.clone();
                result_diff.put_(indices_tensor, diff_source);
            } catch (...) {
                // Type conversion might fail, ignore
            }
        }
        
        // Test boundary conditions
        if (input_tensor.numel() > 0) {
            // Test with maximum valid index
            auto max_idx = torch::tensor({input_tensor.numel() - 1}, torch::kLong);
            auto boundary_source = create_random_tensor({1}, input_dtype, Data, Size, offset);
            auto result_boundary = input_tensor.clone();
            result_boundary.put_(max_idx, boundary_source);
        }
        
        // Verify results are valid tensors
        if (result1.defined()) result1.sum();
        if (result2.defined()) result2.sum();
        if (result3.defined()) result3.sum();
        if (result4.defined()) result4.sum();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}