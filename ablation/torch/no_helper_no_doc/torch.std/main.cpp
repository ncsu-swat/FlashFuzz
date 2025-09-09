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
        
        // Test basic std() without parameters
        auto result1 = torch::std(input);
        
        // Parse dimension parameter for std with dim
        if (offset < Size) {
            int64_t dim = parse_int64(Data, Size, offset);
            if (dim >= -input.dim() && dim < input.dim()) {
                // Test std with dimension
                auto result2 = torch::std(input, dim);
                
                // Parse keepdim parameter
                if (offset < Size) {
                    bool keepdim = parse_bool(Data, Size, offset);
                    auto result3 = torch::std(input, dim, keepdim);
                    
                    // Parse unbiased parameter
                    if (offset < Size) {
                        bool unbiased = parse_bool(Data, Size, offset);
                        auto result4 = torch::std(input, dim, unbiased, keepdim);
                    }
                }
            }
        }
        
        // Test std with multiple dimensions
        if (offset < Size && input.dim() > 1) {
            auto dims = parse_int_list(Data, Size, offset, input.dim());
            if (!dims.empty()) {
                // Validate dimensions are within bounds
                bool valid_dims = true;
                for (auto d : dims) {
                    if (d < -input.dim() || d >= input.dim()) {
                        valid_dims = false;
                        break;
                    }
                }
                
                if (valid_dims) {
                    auto result5 = torch::std(input, dims);
                    
                    if (offset < Size) {
                        bool keepdim = parse_bool(Data, Size, offset);
                        auto result6 = torch::std(input, dims, keepdim);
                        
                        if (offset < Size) {
                            bool unbiased = parse_bool(Data, Size, offset);
                            auto result7 = torch::std(input, dims, unbiased, keepdim);
                        }
                    }
                }
            }
        }
        
        // Test with correction parameter (newer API)
        if (offset < Size) {
            int64_t correction = parse_int64(Data, Size, offset) % 10; // Limit correction value
            if (correction >= 0) {
                auto result8 = torch::std(input, /*dim=*/c10::nullopt, correction);
                
                // Test with dim and correction
                if (offset < Size && input.dim() > 0) {
                    int64_t dim = parse_int64(Data, Size, offset);
                    if (dim >= -input.dim() && dim < input.dim()) {
                        auto result9 = torch::std(input, dim, correction);
                        
                        if (offset < Size) {
                            bool keepdim = parse_bool(Data, Size, offset);
                            auto result10 = torch::std(input, dim, correction, keepdim);
                        }
                    }
                }
            }
        }
        
        // Test edge cases with empty tensors
        if (input.numel() == 0) {
            auto empty_result = torch::std(input);
        }
        
        // Test with complex tensors if applicable
        if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) {
            auto complex_result = torch::std(input);
        }
        
        // Test std_mean function if available
        if (input.numel() > 0) {
            auto std_mean_result = torch::std_mean(input);
            
            if (input.dim() > 0) {
                int64_t dim = 0;
                auto std_mean_dim_result = torch::std_mean(input, dim);
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