#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <algorithm>      // For std::max, std::find

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - ensure it's floating point for std
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch::std requires floating point tensor
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Extract parameters for std operation if we have more data
        bool unbiased = true;  // Default is unbiased (Bessel's correction)
        bool keepdim = false;
        
        if (offset + 1 < Size) {
            unbiased = Data[offset++] & 0x1;
        }
        
        if (offset + 1 < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Variant 1: std over all elements (returns scalar tensor)
        // torch::std(input) computes std over all elements
        try {
            torch::Tensor result1 = torch::std(input_tensor);
            (void)result1;
        } catch (const std::exception&) {
            // Silently catch expected failures
        }
        
        // Variant 2: std with unbiased correction flag
        try {
            torch::Tensor result2 = torch::std(input_tensor, unbiased);
            (void)result2;
        } catch (const std::exception&) {
            // Silently catch expected failures
        }
        
        // Variant 3: std along specific dimension if tensor has dimensions
        if (input_tensor.dim() > 0 && offset < Size) {
            // Get a dimension to reduce along
            int64_t dim = static_cast<int64_t>(Data[offset++]) % std::max(static_cast<int64_t>(1), input_tensor.dim());
            
            // Try negative dimension index too
            if (offset < Size && (Data[offset++] & 0x1)) {
                dim = -dim - 1;
            }
            
            // std along dimension with unbiased and keepdim
            try {
                torch::Tensor result3 = torch::std(input_tensor, dim, unbiased, keepdim);
                (void)result3;
            } catch (const std::exception&) {
                // Silently catch expected failures
            }
            
            // Try with IntArrayRef (list of dimensions) if tensor has multiple dimensions
            if (input_tensor.dim() > 1 && offset < Size) {
                std::vector<int64_t> dims;
                uint8_t num_dims = (Data[offset++] % input_tensor.dim()) + 1;
                
                for (uint8_t i = 0; i < num_dims && offset < Size && static_cast<int64_t>(dims.size()) < input_tensor.dim(); i++) {
                    int64_t d = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
                    
                    // Ensure no duplicate dimensions
                    if (std::find(dims.begin(), dims.end(), d) == dims.end()) {
                        dims.push_back(d);
                    }
                }
                
                if (!dims.empty()) {
                    // std along multiple dimensions
                    try {
                        torch::Tensor result4 = torch::std(input_tensor, dims, unbiased, keepdim);
                        (void)result4;
                    } catch (const std::exception&) {
                        // Silently catch expected failures
                    }
                }
            }
        }
        
        // Variant with correction parameter (newer API)
        // torch::std(input, dim, correction, keepdim)
        if (offset + 1 < Size && input_tensor.dim() > 0) {
            int64_t correction = static_cast<int64_t>(Data[offset++] % 3);  // 0, 1, or 2
            int64_t dim = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
            
            try {
                // Use std with optional correction
                // The c10::optional<int64_t> version: std(input, dims, correction, keepdim)
                std::vector<int64_t> dims_vec = {dim};
                torch::Tensor result5 = torch::std(input_tensor, dims_vec, correction, keepdim);
                (void)result5;
            } catch (const std::exception&) {
                // Silently catch expected failures
            }
        }
        
        // Test with empty dimension list (reduces all dimensions)
        try {
            std::vector<int64_t> all_dims;
            for (int64_t i = 0; i < input_tensor.dim(); i++) {
                all_dims.push_back(i);
            }
            if (!all_dims.empty()) {
                torch::Tensor result6 = torch::std(input_tensor, all_dims, unbiased, keepdim);
                (void)result6;
            }
        } catch (const std::exception&) {
            // Silently catch expected failures
        }
        
        // Test std_mean which returns both std and mean
        if (input_tensor.dim() > 0 && offset < Size) {
            int64_t dim = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
            try {
                auto [std_result, mean_result] = torch::std_mean(input_tensor, dim, unbiased, keepdim);
                (void)std_result;
                (void)mean_result;
            } catch (const std::exception&) {
                // Silently catch expected failures
            }
        }
        
        // Test std_mean without dimension (over all elements)
        try {
            auto [std_result, mean_result] = torch::std_mean(input_tensor, unbiased);
            (void)std_result;
            (void)mean_result;
        } catch (const std::exception&) {
            // Silently catch expected failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}