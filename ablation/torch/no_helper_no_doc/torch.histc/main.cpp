#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some bytes for basic parameters
        if (Size < 16) {
            return 0;
        }

        // Extract basic parameters
        int64_t bins = extract_int64_t(Data, Size, offset) % 1000 + 1; // 1 to 1000 bins
        double min_val = extract_double(Data, Size, offset);
        double max_val = extract_double(Data, Size, offset);
        
        // Ensure min < max, swap if necessary
        if (min_val > max_val) {
            std::swap(min_val, max_val);
        }
        
        // If they're equal, add small difference
        if (min_val == max_val) {
            max_val = min_val + 1.0;
        }

        // Create input tensor with various shapes and data types
        auto tensor_info = extract_tensor_info(Data, Size, offset);
        
        // Limit tensor size to avoid memory issues
        for (auto& dim : tensor_info.shape) {
            dim = std::abs(dim) % 100 + 1; // 1 to 100 per dimension
        }
        
        // Test with different data types that make sense for histc
        std::vector<torch::ScalarType> valid_types = {
            torch::kFloat32, torch::kFloat64, torch::kInt32, torch::kInt64
        };
        
        torch::ScalarType dtype = valid_types[extract_uint8_t(Data, Size, offset) % valid_types.size()];
        
        // Create tensor
        torch::Tensor input;
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            input = torch::randn(tensor_info.shape, torch::TensorOptions().dtype(dtype));
            // Add some values in the range [min_val, max_val]
            input = input * (max_val - min_val) + min_val;
        } else {
            input = torch::randint(static_cast<int64_t>(min_val), 
                                 static_cast<int64_t>(max_val) + 1, 
                                 tensor_info.shape, 
                                 torch::TensorOptions().dtype(dtype));
        }

        // Test basic histc call
        auto result1 = torch::histc(input, bins, min_val, max_val);
        
        // Test with different bins values
        int64_t bins2 = extract_int64_t(Data, Size, offset) % 500 + 1;
        auto result2 = torch::histc(input, bins2, min_val, max_val);
        
        // Test with edge case: single bin
        auto result3 = torch::histc(input, 1, min_val, max_val);
        
        // Test with very small range
        double small_range_min = min_val;
        double small_range_max = min_val + 1e-6;
        auto result4 = torch::histc(input, bins, small_range_min, small_range_max);
        
        // Test with negative ranges
        double neg_min = -std::abs(min_val) - 10.0;
        double neg_max = -std::abs(min_val);
        auto result5 = torch::histc(input, bins, neg_min, neg_max);
        
        // Test with large ranges
        double large_min = -1e6;
        double large_max = 1e6;
        auto result6 = torch::histc(input, bins, large_min, large_max);
        
        // Test with empty tensor (if possible)
        if (input.numel() > 0) {
            torch::Tensor empty_input = torch::empty({0}, input.options());
            auto result7 = torch::histc(empty_input, bins, min_val, max_val);
        }
        
        // Test with 1D tensor
        torch::Tensor input_1d = input.flatten();
        auto result8 = torch::histc(input_1d, bins, min_val, max_val);
        
        // Test with tensor containing special values (if float type)
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            torch::Tensor special_input = input.clone();
            if (special_input.numel() > 0) {
                // Add some inf, -inf, nan values
                special_input.flatten()[0] = std::numeric_limits<double>::infinity();
                if (special_input.numel() > 1) {
                    special_input.flatten()[1] = -std::numeric_limits<double>::infinity();
                }
                if (special_input.numel() > 2) {
                    special_input.flatten()[2] = std::numeric_limits<double>::quiet_NaN();
                }
                auto result9 = torch::histc(special_input, bins, min_val, max_val);
            }
        }
        
        // Test with different tensor layouts/strides
        if (input.dim() > 1) {
            auto transposed = input.transpose(0, 1);
            auto result10 = torch::histc(transposed, bins, min_val, max_val);
        }
        
        // Test with contiguous and non-contiguous tensors
        if (input.dim() > 1 && input.size(0) > 1) {
            auto sliced = input.slice(0, 0, input.size(0), 2); // every other element
            auto result11 = torch::histc(sliced, bins, min_val, max_val);
        }
        
        // Verify result properties
        if (result1.defined()) {
            // Result should be 1D tensor with 'bins' elements
            assert(result1.dim() == 1);
            assert(result1.size(0) == bins);
            assert(result1.dtype() == torch::kFloat32 || result1.dtype() == torch::kFloat64);
            
            // All values should be non-negative (counts)
            auto min_count = torch::min(result1);
            assert(min_count.item<double>() >= 0.0);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}