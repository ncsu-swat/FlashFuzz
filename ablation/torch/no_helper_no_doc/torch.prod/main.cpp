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
        auto input = create_tensor(Data, Size, offset, shape, dtype);
        if (!input.defined()) return 0;
        
        // Test 1: Basic prod() without arguments
        auto result1 = torch::prod(input);
        
        // Test 2: prod() with dim argument
        if (input.dim() > 0) {
            int64_t dim = parse_int64_t(Data, Size, offset) % input.dim();
            auto result2 = torch::prod(input, dim);
            
            // Test 3: prod() with dim and keepdim
            bool keepdim = parse_bool(Data, Size, offset);
            auto result3 = torch::prod(input, dim, keepdim);
            
            // Test 4: prod() with dim, keepdim and dtype
            auto out_dtype = parse_dtype(Data, Size, offset);
            auto result4 = torch::prod(input, dim, keepdim, out_dtype);
        }
        
        // Test 5: prod() with only dtype argument
        auto out_dtype2 = parse_dtype(Data, Size, offset);
        auto result5 = torch::prod(input, out_dtype2);
        
        // Test 6: In-place operations if tensor allows
        if (input.is_floating_point() || input.is_complex()) {
            auto input_copy = input.clone();
            if (input_copy.dim() > 0) {
                int64_t dim = parse_int64_t(Data, Size, offset) % input_copy.dim();
                bool keepdim = parse_bool(Data, Size, offset);
                
                // Create output tensor for in-place operation
                auto out_shape = input_copy.sizes().vec();
                if (!keepdim && input_copy.dim() > 0) {
                    out_shape.erase(out_shape.begin() + dim);
                } else if (keepdim && input_copy.dim() > 0) {
                    out_shape[dim] = 1;
                }
                
                if (!out_shape.empty()) {
                    auto out_tensor = torch::empty(out_shape, input_copy.options());
                    torch::prod_out(out_tensor, input_copy, dim, keepdim);
                }
            }
        }
        
        // Test 7: Edge cases with different tensor properties
        if (input.numel() > 0) {
            // Test with negative dimensions
            if (input.dim() > 0) {
                int64_t neg_dim = -(parse_int64_t(Data, Size, offset) % input.dim() + 1);
                auto result6 = torch::prod(input, neg_dim);
            }
            
            // Test with scalar tensor
            if (input.numel() == 1) {
                auto scalar_result = torch::prod(input.view({}));
            }
        }
        
        // Test 8: Different data types and edge values
        if (input.is_floating_point()) {
            // Test with tensor containing special values
            auto special_tensor = input.clone();
            if (special_tensor.numel() > 0) {
                // Set some elements to special values if possible
                auto flat = special_tensor.flatten();
                if (flat.numel() > 0) {
                    if (parse_bool(Data, Size, offset)) {
                        flat[0] = std::numeric_limits<float>::infinity();
                    }
                    if (flat.numel() > 1 && parse_bool(Data, Size, offset)) {
                        flat[1] = -std::numeric_limits<float>::infinity();
                    }
                    if (flat.numel() > 2 && parse_bool(Data, Size, offset)) {
                        flat[2] = std::numeric_limits<float>::quiet_NaN();
                    }
                }
                auto special_result = torch::prod(special_tensor);
            }
        }
        
        // Test 9: Large dimension index (should wrap around)
        if (input.dim() > 0) {
            int64_t large_dim = parse_int64_t(Data, Size, offset);
            // This should wrap around due to modulo operation in PyTorch
            auto result7 = torch::prod(input, large_dim % input.dim());
        }
        
        // Test 10: Empty tensor cases
        if (input.numel() == 0) {
            auto empty_result = torch::prod(input);
            if (input.dim() > 0) {
                int64_t dim = parse_int64_t(Data, Size, offset) % input.dim();
                auto empty_dim_result = torch::prod(input, dim);
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}