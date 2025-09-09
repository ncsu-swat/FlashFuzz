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
        
        // Parse optional parameters
        auto dim = parse_optional_dim_list(Data, Size, offset, input.dim());
        bool unbiased = parse_bool(Data, Size, offset);
        bool keepdim = parse_bool(Data, Size, offset);
        
        // Test var_mean with different parameter combinations
        
        // Case 1: Basic var_mean without parameters
        auto result1 = torch::var_mean(input);
        auto var1 = std::get<0>(result1);
        auto mean1 = std::get<1>(result1);
        
        // Case 2: var_mean with unbiased parameter
        auto result2 = torch::var_mean(input, unbiased);
        auto var2 = std::get<0>(result2);
        auto mean2 = std::get<1>(result2);
        
        // Case 3: var_mean with dim parameter (if valid)
        if (dim.has_value() && !dim.value().empty()) {
            auto result3 = torch::var_mean(input, dim.value());
            auto var3 = std::get<0>(result3);
            auto mean3 = std::get<1>(result3);
            
            // Case 4: var_mean with dim and unbiased
            auto result4 = torch::var_mean(input, dim.value(), unbiased);
            auto var4 = std::get<0>(result4);
            auto mean4 = std::get<1>(result4);
            
            // Case 5: var_mean with dim, unbiased, and keepdim
            auto result5 = torch::var_mean(input, dim.value(), unbiased, keepdim);
            auto var5 = std::get<0>(result5);
            auto mean5 = std::get<1>(result5);
        }
        
        // Test with different correction values if we have enough data
        if (offset < Size) {
            int64_t correction = parse_int64(Data, Size, offset) % 10; // Limit correction range
            
            if (dim.has_value() && !dim.value().empty()) {
                auto result6 = torch::var_mean(input, dim.value(), correction, keepdim);
                auto var6 = std::get<0>(result6);
                auto mean6 = std::get<1>(result6);
            }
        }
        
        // Test edge cases with specific tensor configurations
        if (input.numel() > 0) {
            // Test with all dimensions
            std::vector<int64_t> all_dims;
            for (int64_t i = 0; i < input.dim(); ++i) {
                all_dims.push_back(i);
            }
            if (!all_dims.empty()) {
                auto result_all = torch::var_mean(input, all_dims, unbiased, keepdim);
                auto var_all = std::get<0>(result_all);
                auto mean_all = std::get<1>(result_all);
            }
            
            // Test with single dimension
            if (input.dim() > 0) {
                int64_t single_dim = parse_int64(Data, Size, offset) % input.dim();
                auto result_single = torch::var_mean(input, {single_dim}, unbiased, keepdim);
                auto var_single = std::get<0>(result_single);
                auto mean_single = std::get<1>(result_single);
            }
        }
        
        // Test with complex tensors if supported
        if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) {
            auto result_complex = torch::var_mean(input, unbiased);
            auto var_complex = std::get<0>(result_complex);
            auto mean_complex = std::get<1>(result_complex);
        }
        
        // Test with very small tensors
        if (input.numel() == 1) {
            auto result_scalar = torch::var_mean(input);
            auto var_scalar = std::get<0>(result_scalar);
            auto mean_scalar = std::get<1>(result_scalar);
        }
        
        // Verify results are finite when possible
        if (var1.defined() && var1.numel() > 0) {
            torch::isfinite(var1);
        }
        if (mean1.defined() && mean1.numel() > 0) {
            torch::isfinite(mean1);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}