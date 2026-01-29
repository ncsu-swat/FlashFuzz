#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <algorithm>      // For std::max, std::find

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - needs to be floating point for variance
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if not already a floating point type
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Extract parameters for var operation if we have more data
        bool unbiased = true;  // Default is unbiased (Bessel's correction)
        bool keepdim = false;
        
        if (offset + 1 <= Size) {
            unbiased = Data[offset++] & 0x1;
        }
        
        if (offset + 1 <= Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Variant 1: var over all dimensions (returns scalar)
        try {
            torch::Tensor result1 = torch::var(input_tensor, unbiased);
            (void)result1;
        } catch (const std::exception&) {
            // Silently catch expected failures
        }
        
        // Variant 2: var along specific dimension if tensor has dimensions
        if (input_tensor.dim() > 0 && offset < Size) {
            // Get a dimension to reduce along
            int64_t dim = static_cast<int64_t>(Data[offset++]) % std::max(static_cast<int64_t>(1), input_tensor.dim());
            
            // Try negative dimension index sometimes
            if (offset < Size && (Data[offset++] & 0x1)) {
                dim = -dim - 1;
            }
            
            try {
                torch::Tensor result2 = torch::var(input_tensor, dim, unbiased, keepdim);
                (void)result2;
            } catch (const std::exception&) {
                // Silently catch expected failures
            }
        }
        
        // Variant 3: var with a list of dimensions if tensor has multiple dimensions
        if (input_tensor.dim() > 1 && offset < Size) {
            int num_dims = (Data[offset++] % (input_tensor.dim() - 1)) + 1;
            std::vector<int64_t> dims;
            
            for (int i = 0; i < num_dims && offset < Size; i++) {
                int64_t d = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
                // Ensure no duplicate dimensions
                if (std::find(dims.begin(), dims.end(), d) == dims.end()) {
                    dims.push_back(d);
                }
            }
            
            if (!dims.empty()) {
                try {
                    torch::Tensor result3 = torch::var(input_tensor, dims, unbiased, keepdim);
                    (void)result3;
                } catch (const std::exception&) {
                    // Silently catch expected failures
                }
            }
        }
        
        // Variant 4: var with correction parameter (c10::optional<int64_t>)
        // correction=1 is Bessel's correction (unbiased), correction=0 is biased
        if (offset < Size) {
            int64_t correction = static_cast<int64_t>(Data[offset++]) % 3;  // 0, 1, or 2
            
            try {
                // var with empty dims (reduce all), correction, and keepdim
                torch::Tensor result4 = torch::var(input_tensor, c10::IntArrayRef{}, correction, keepdim);
                (void)result4;
            } catch (const std::exception&) {
                // Silently catch expected failures
            }
            
            // Also try with specific dimension and correction
            if (input_tensor.dim() > 0 && offset < Size) {
                int64_t dim = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
                std::vector<int64_t> dim_vec = {dim};
                
                try {
                    torch::Tensor result5 = torch::var(input_tensor, dim_vec, correction, keepdim);
                    (void)result5;
                } catch (const std::exception&) {
                    // Silently catch expected failures
                }
            }
        }
        
        // Variant 5: var_mean - returns both variance and mean
        try {
            auto result_tuple = torch::var_mean(input_tensor, unbiased);
            torch::Tensor var_result = std::get<0>(result_tuple);
            torch::Tensor mean_result = std::get<1>(result_tuple);
            (void)var_result;
            (void)mean_result;
        } catch (const std::exception&) {
            // Silently catch expected failures
        }
        
        // Variant 6: var_mean along dimension
        if (input_tensor.dim() > 0 && offset < Size) {
            int64_t dim = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
            
            try {
                auto result_tuple = torch::var_mean(input_tensor, dim, unbiased, keepdim);
                torch::Tensor var_result = std::get<0>(result_tuple);
                torch::Tensor mean_result = std::get<1>(result_tuple);
                (void)var_result;
                (void)mean_result;
            } catch (const std::exception&) {
                // Silently catch expected failures
            }
        }
        
        // Variant 7: Test with different tensor sizes and types
        if (offset + 2 < Size) {
            int64_t size1 = (Data[offset++] % 10) + 1;
            int64_t size2 = (Data[offset++] % 10) + 1;
            
            torch::Tensor test_tensor = torch::randn({size1, size2});
            
            try {
                torch::Tensor result6 = torch::var(test_tensor, 0, unbiased, keepdim);
                torch::Tensor result7 = torch::var(test_tensor, 1, unbiased, keepdim);
                (void)result6;
                (void)result7;
            } catch (const std::exception&) {
                // Silently catch expected failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}