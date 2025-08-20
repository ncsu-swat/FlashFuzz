#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for var_mean from the remaining data
        bool unbiased = false;
        if (offset < Size) {
            unbiased = Data[offset++] & 0x1;
        }
        
        // Get a dimension to compute var_mean along
        int64_t dim = 0;
        if (input_tensor.dim() > 0 && offset < Size) {
            dim = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
        }
        
        // Get keepdim parameter
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Try different variants of var_mean
        
        // Variant 1: var_mean on entire tensor
        auto result1 = torch::var_mean(input_tensor, unbiased);
        
        // Variant 2: var_mean along a dimension
        if (input_tensor.dim() > 0) {
            auto result2 = torch::var_mean(input_tensor, dim, unbiased, keepdim);
            
            // Access the results to ensure they're computed
            torch::Tensor var = std::get<0>(result2);
            torch::Tensor mean = std::get<1>(result2);
        }
        
        // Variant 3: var_mean along multiple dimensions (if tensor has at least 2 dimensions)
        if (input_tensor.dim() >= 2 && offset + 1 < Size) {
            int64_t dim2 = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
            if (dim2 == dim) {
                dim2 = (dim2 + 1) % input_tensor.dim();
            }
            
            std::vector<int64_t> dims = {dim, dim2};
            auto result3 = torch::var_mean(input_tensor, dims, unbiased, keepdim);
        }
        
        // Variant 4: var_mean with int dim parameter
        if (input_tensor.dim() > 0) {
            int int_dim = static_cast<int>(dim);
            auto result4 = torch::var_mean(input_tensor, int_dim);
            
            auto result5 = torch::var_mean(input_tensor, int_dim, unbiased, keepdim);
        }
        
        // Variant 5: Named dimension variant (if available)
        if (input_tensor.dim() > 0) {
            auto result6 = torch::var_mean(input_tensor, {dim}, unbiased, keepdim);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}