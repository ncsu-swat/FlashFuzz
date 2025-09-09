#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse basic parameters
        if (Size < 20) return 0; // Need minimum data for parameters
        
        // Extract dimensions and parameters
        int batch_size = std::max(1, static_cast<int>(Data[offset++] % 4 + 1)); // 1-4
        int m = std::max(1, static_cast<int>(Data[offset++] % 8 + 1)); // 1-8
        int n = std::max(1, static_cast<int>(Data[offset++] % 8 + 1)); // 1-8
        int k = std::max(1, static_cast<int>(Data[offset++] % std::min(m, n) + 1)); // 1-min(m,n)
        
        bool left = (Data[offset++] % 2) == 0;
        bool transpose = (Data[offset++] % 2) == 0;
        
        // Determine dtype
        int dtype_choice = Data[offset++] % 4;
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat; break;
            case 1: dtype = torch::kDouble; break;
            case 2: dtype = torch::kComplexFloat; break;
            case 3: dtype = torch::kComplexDouble; break;
        }
        
        // Calculate tensor dimensions based on left parameter
        int mn = left ? m : n;
        int min_mn_k = std::min(mn, k);
        
        // Create input tensor (Householder reflectors)
        // Shape: (batch_size, mn, k)
        torch::Tensor input;
        if (batch_size > 1) {
            input = torch::randn({batch_size, mn, k}, torch::TensorOptions().dtype(dtype));
        } else {
            input = torch::randn({mn, k}, torch::TensorOptions().dtype(dtype));
        }
        
        // Create tau tensor (scaling factors for Householder reflectors)
        // Shape: (batch_size, min(mn, k))
        torch::Tensor tau;
        if (batch_size > 1) {
            tau = torch::randn({batch_size, min_mn_k}, torch::TensorOptions().dtype(dtype));
        } else {
            tau = torch::randn({min_mn_k}, torch::TensorOptions().dtype(dtype));
        }
        
        // Create other tensor (matrix to multiply with Q)
        // Shape: (batch_size, m, n)
        torch::Tensor other;
        if (batch_size > 1) {
            other = torch::randn({batch_size, m, n}, torch::TensorOptions().dtype(dtype));
        } else {
            other = torch::randn({m, n}, torch::TensorOptions().dtype(dtype));
        }
        
        // Add some noise to tensors based on remaining data
        if (offset < Size) {
            auto noise_scale = static_cast<double>(Data[offset++] % 100) / 100.0;
            input = input + noise_scale * torch::randn_like(input);
            tau = tau + noise_scale * torch::randn_like(tau);
            other = other + noise_scale * torch::randn_like(other);
        }
        
        // Test edge cases with special values
        if (offset < Size) {
            int special_case = Data[offset++] % 6;
            switch (special_case) {
                case 0: // Zero tensors
                    input = torch::zeros_like(input);
                    break;
                case 1: // Very small values
                    input = input * 1e-10;
                    tau = tau * 1e-10;
                    break;
                case 2: // Very large values
                    input = input * 1e10;
                    tau = tau * 1e10;
                    break;
                case 3: // Mixed signs
                    input = torch::abs(input) * torch::sign(torch::randn_like(input));
                    break;
                case 4: // Identity-like tau
                    tau = torch::ones_like(tau);
                    break;
                case 5: // NaN/Inf handling (but avoid actual NaN/Inf)
                    input = torch::clamp(input, -1e6, 1e6);
                    tau = torch::clamp(tau, -1e6, 1e6);
                    break;
            }
        }
        
        // Call torch.ormqr with different parameter combinations
        torch::Tensor result1 = torch::ormqr(input, tau, other, left, transpose);
        
        // Test with opposite boolean parameters
        torch::Tensor result2 = torch::ormqr(input, tau, other, !left, transpose);
        torch::Tensor result3 = torch::ormqr(input, tau, other, left, !transpose);
        torch::Tensor result4 = torch::ormqr(input, tau, other, !left, !transpose);
        
        // Test with output tensor
        if (offset < Size && (Data[offset++] % 2) == 0) {
            torch::Tensor out = torch::empty_like(other);
            torch::ormqr_out(out, input, tau, other, left, transpose);
        }
        
        // Verify results have expected shapes
        auto expected_shape = other.sizes();
        if (result1.sizes() != expected_shape) {
            throw std::runtime_error("Result shape mismatch");
        }
        
        // Test with different tensor layouts/strides if there's remaining data
        if (offset < Size) {
            int layout_test = Data[offset++] % 3;
            switch (layout_test) {
                case 0: {
                    // Test with transposed input
                    auto input_t = input.transpose(-2, -1).contiguous().transpose(-2, -1);
                    torch::ormqr(input_t, tau, other, left, transpose);
                    break;
                }
                case 1: {
                    // Test with non-contiguous other
                    if (other.dim() >= 2) {
                        auto other_nc = other.transpose(-2, -1).contiguous().transpose(-2, -1);
                        torch::ormqr(input, tau, other_nc, left, transpose);
                    }
                    break;
                }
                case 2: {
                    // Test with sliced tensors
                    if (input.size(-1) > 1) {
                        auto input_slice = input.narrow(-1, 0, input.size(-1) - 1);
                        auto tau_slice = tau.narrow(-1, 0, std::min(tau.size(-1), input_slice.size(-1)));
                        torch::ormqr(input_slice, tau_slice, other, left, transpose);
                    }
                    break;
                }
            }
        }
        
        // Test gradient computation if there's remaining data
        if (offset < Size && (Data[offset++] % 4) == 0) {
            input.requires_grad_(true);
            tau.requires_grad_(true);
            other.requires_grad_(true);
            
            auto result_grad = torch::ormqr(input, tau, other, left, transpose);
            auto loss = result_grad.sum();
            loss.backward();
        }
        
        // Test with different device if CUDA is available
        if (torch::cuda::is_available() && offset < Size && (Data[offset++] % 8) == 0) {
            auto input_cuda = input.to(torch::kCUDA);
            auto tau_cuda = tau.to(torch::kCUDA);
            auto other_cuda = other.to(torch::kCUDA);
            torch::ormqr(input_cuda, tau_cuda, other_cuda, left, transpose);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}