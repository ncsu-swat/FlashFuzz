#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor dimensions and data type
        auto dims = parse_tensor_dims(Data, Size, offset);
        if (dims.empty()) return 0;
        
        auto dtype = parse_dtype(Data, Size, offset);
        
        // Create input tensor
        auto input = create_tensor(dims, dtype, Data, Size, offset);
        if (!input.defined()) return 0;
        
        // Test case 1: amin() without any parameters (global minimum)
        auto result1 = torch::amin(input);
        
        // Test case 2: amin with dim parameter
        if (input.dim() > 0) {
            // Parse dimension to reduce along
            int64_t dim = parse_int_in_range(Data, Size, offset, -input.dim(), input.dim() - 1);
            auto result2 = torch::amin(input, dim);
            
            // Test case 3: amin with dim and keepdim=true
            auto result3 = torch::amin(input, dim, true);
            
            // Test case 4: amin with multiple dimensions
            if (input.dim() > 1) {
                std::vector<int64_t> dims_vec;
                int num_dims = parse_int_in_range(Data, Size, offset, 1, std::min(3, (int)input.dim()));
                for (int i = 0; i < num_dims; i++) {
                    int64_t d = parse_int_in_range(Data, Size, offset, -input.dim(), input.dim() - 1);
                    // Avoid duplicate dimensions
                    if (std::find(dims_vec.begin(), dims_vec.end(), d) == dims_vec.end()) {
                        dims_vec.push_back(d);
                    }
                }
                
                if (!dims_vec.empty()) {
                    auto result4 = torch::amin(input, dims_vec);
                    auto result5 = torch::amin(input, dims_vec, true);
                }
            }
        }
        
        // Test edge cases with special values if floating point
        if (input.dtype().is_floating_point()) {
            // Create tensor with special values
            auto special_tensor = input.clone();
            if (special_tensor.numel() > 0) {
                // Add some inf, -inf, nan values
                auto flat = special_tensor.flatten();
                if (flat.numel() >= 3) {
                    flat[0] = std::numeric_limits<float>::infinity();
                    flat[1] = -std::numeric_limits<float>::infinity();
                    flat[2] = std::numeric_limits<float>::quiet_NaN();
                }
                
                auto special_result = torch::amin(special_tensor);
                
                if (special_tensor.dim() > 0) {
                    auto special_result_dim = torch::amin(special_tensor, 0);
                }
            }
        }
        
        // Test with empty tensor
        if (input.numel() == 0 && input.dim() > 0) {
            try {
                auto empty_result = torch::amin(input);
            } catch (...) {
                // Expected to potentially fail with empty tensors
            }
        }
        
        // Test with very large/small dimensions
        if (input.dim() > 0) {
            try {
                // Test with out-of-bounds dimension (should throw)
                auto invalid_result = torch::amin(input, input.dim() + 10);
            } catch (...) {
                // Expected to fail
            }
            
            try {
                // Test with negative out-of-bounds dimension
                auto invalid_result2 = torch::amin(input, -(input.dim() + 10));
            } catch (...) {
                // Expected to fail
            }
        }
        
        // Test with different tensor layouts if possible
        if (input.dim() >= 2 && input.numel() > 1) {
            try {
                auto transposed = input.transpose(0, 1);
                auto transposed_result = torch::amin(transposed);
            } catch (...) {
                // May fail for some tensor configurations
            }
        }
        
        // Test with contiguous and non-contiguous tensors
        if (input.dim() > 1) {
            try {
                auto sliced = input.slice(0, 0, -1, 2); // Every other element
                if (sliced.numel() > 0) {
                    auto sliced_result = torch::amin(sliced);
                }
            } catch (...) {
                // May fail for some configurations
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