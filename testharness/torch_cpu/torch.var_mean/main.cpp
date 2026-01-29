#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        
        // Create input tensor - needs to be floating point for var_mean
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // var_mean requires floating point tensor
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Skip empty tensors
        if (input_tensor.numel() == 0) {
            return 0;
        }
        
        // Extract parameters from the remaining data
        int64_t correction = 1; // default: unbiased (Bessel's correction)
        if (offset < Size) {
            correction = Data[offset++] % 3; // 0, 1, or 2
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
        
        // Variant 1: var_mean on entire tensor (returns tuple of scalars)
        {
            auto result = torch::var_mean(input_tensor);
            torch::Tensor var = std::get<0>(result);
            torch::Tensor mean = std::get<1>(result);
            // Force computation
            (void)var.item<float>();
            (void)mean.item<float>();
        }
        
        // Variant 2: var_mean along a dimension
        if (input_tensor.dim() > 0) {
            try {
                auto result = torch::var_mean(input_tensor, {dim}, correction, keepdim);
                torch::Tensor var = std::get<0>(result);
                torch::Tensor mean = std::get<1>(result);
                // Access results to ensure computation
                (void)var.numel();
                (void)mean.numel();
            } catch (const std::exception &) {
                // Some dimension combinations may be invalid, ignore
            }
        }
        
        // Variant 3: var_mean along multiple dimensions
        if (input_tensor.dim() >= 2 && offset < Size) {
            try {
                int64_t dim2 = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
                // Ensure different dimensions
                if (dim2 == dim) {
                    dim2 = (dim2 + 1) % input_tensor.dim();
                }
                
                std::vector<int64_t> dims = {std::min(dim, dim2), std::max(dim, dim2)};
                auto result = torch::var_mean(input_tensor, dims, correction, keepdim);
                torch::Tensor var = std::get<0>(result);
                torch::Tensor mean = std::get<1>(result);
                (void)var.numel();
                (void)mean.numel();
            } catch (const std::exception &) {
                // Invalid dimension combinations, ignore
            }
        }
        
        // Variant 4: var_mean with different correction values
        if (input_tensor.dim() > 0 && input_tensor.size(dim) > 2) {
            try {
                // correction = 0 (biased/population variance)
                auto result_biased = torch::var_mean(input_tensor, {dim}, 0, keepdim);
                (void)std::get<0>(result_biased).numel();
                
                // correction = 2 (higher correction)
                auto result_corr2 = torch::var_mean(input_tensor, {dim}, 2, keepdim);
                (void)std::get<0>(result_corr2).numel();
            } catch (const std::exception &) {
                // May fail if size along dim <= correction
            }
        }
        
        // Variant 5: Test with different dtypes
        if (offset < Size) {
            try {
                torch::Tensor double_tensor = input_tensor.to(torch::kFloat64);
                auto result = torch::var_mean(double_tensor);
                (void)std::get<0>(result).item<double>();
            } catch (const std::exception &) {
                // Dtype conversion issues, ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}