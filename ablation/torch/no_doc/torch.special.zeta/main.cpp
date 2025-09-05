#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for creating two tensors
        if (Size < 4) {
            // Not enough data to create even minimal tensors
            return 0;
        }

        // Create first tensor (x argument for zeta)
        torch::Tensor x;
        try {
            x = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception &e) {
            // If we can't create the first tensor, try with minimal tensor
            x = torch::ones({1});
            offset = Size / 2; // Move offset to middle for second tensor
        }

        // Create second tensor (q argument for zeta)
        torch::Tensor q;
        try {
            q = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception &e) {
            // If we can't create the second tensor, create a minimal one
            q = torch::ones({1});
        }

        // Ensure tensors have compatible dtypes for zeta
        // zeta typically works with floating point types
        if (!x.is_floating_point()) {
            x = x.to(torch::kFloat32);
        }
        if (!q.is_floating_point()) {
            q = q.to(torch::kFloat32);
        }

        // Test various broadcasting scenarios by potentially reshaping
        if (offset < Size) {
            uint8_t reshape_flag = Data[offset % Size];
            
            // Try different shape combinations for broadcasting tests
            if (reshape_flag & 0x01) {
                // Make x scalar
                if (x.numel() > 0) {
                    x = x.flatten()[0];
                }
            }
            if (reshape_flag & 0x02) {
                // Make q scalar
                if (q.numel() > 0) {
                    q = q.flatten()[0];
                }
            }
            if (reshape_flag & 0x04) {
                // Try to make them broadcastable with different shapes
                if (x.dim() > 0 && q.dim() > 0) {
                    auto x_size = x.numel();
                    auto q_size = q.numel();
                    if (x_size > 1 && q_size > 1) {
                        x = x.view({-1, 1});
                        q = q.view({1, -1});
                    }
                }
            }
        }

        // Main operation: torch.special.zeta
        torch::Tensor result;
        try {
            result = torch::special::zeta(x, q);
            
            // Verify result properties
            if (result.defined()) {
                // Check for NaN/Inf
                auto has_nan = torch::any(torch::isnan(result));
                auto has_inf = torch::any(torch::isinf(result));
                
                // Access the values to ensure computation completed
                if (result.numel() > 0 && result.numel() < 100) {
                    // For small tensors, force evaluation
                    auto sum = result.sum();
                    (void)sum;
                }
            }
        } catch (const c10::Error &e) {
            // PyTorch-specific errors are expected for invalid inputs
            // Continue fuzzing
            return 0;
        }

        // Test with edge case values if we have more data
        if (offset + 1 < Size) {
            uint8_t edge_case = Data[offset % Size];
            
            // Create edge case tensors
            torch::Tensor x_edge, q_edge;
            
            switch (edge_case % 8) {
                case 0:
                    // Test with zeros (may cause singularities)
                    x_edge = torch::zeros_like(x);
                    q_edge = q.clone();
                    break;
                case 1:
                    // Test with ones
                    x_edge = torch::ones_like(x);
                    q_edge = torch::ones_like(q);
                    break;
                case 2:
                    // Test with negative values
                    x_edge = -torch::abs(x);
                    q_edge = -torch::abs(q);
                    break;
                case 3:
                    // Test with very small values
                    x_edge = x * 1e-10;
                    q_edge = q * 1e-10;
                    break;
                case 4:
                    // Test with very large values
                    x_edge = x * 1e10;
                    q_edge = q * 1e10;
                    break;
                case 5:
                    // Test with infinity
                    x_edge = torch::full_like(x, std::numeric_limits<float>::infinity());
                    q_edge = q.clone();
                    break;
                case 6:
                    // Test with NaN
                    x_edge = torch::full_like(x, std::numeric_limits<float>::quiet_NaN());
                    q_edge = q.clone();
                    break;
                case 7:
                    // Test mixed positive/negative
                    x_edge = x - x.mean();
                    q_edge = q - q.mean();
                    break;
                default:
                    x_edge = x.clone();
                    q_edge = q.clone();
            }
            
            try {
                auto edge_result = torch::special::zeta(x_edge, q_edge);
                (void)edge_result;
            } catch (const c10::Error &e) {
                // Expected for some edge cases
            }
        }

        // Test in-place operations and gradient computation if applicable
        if (x.requires_grad() || q.requires_grad()) {
            try {
                x.set_requires_grad(true);
                q.set_requires_grad(true);
                auto result_grad = torch::special::zeta(x, q);
                if (result_grad.numel() > 0) {
                    auto loss = result_grad.sum();
                    loss.backward();
                }
            } catch (const c10::Error &e) {
                // Gradient computation might fail for certain inputs
            }
        }

        // Test with different memory layouts
        if (x.numel() > 1 && x.dim() > 1) {
            try {
                auto x_transposed = x.transpose(0, -1);
                auto result_transposed = torch::special::zeta(x_transposed, q);
                (void)result_transposed;
            } catch (const c10::Error &e) {
                // Some layouts might not be supported
            }
        }

        // Test with different devices if available
        if (torch::cuda::is_available() && (offset % 4 == 0)) {
            try {
                auto x_cuda = x.to(torch::kCUDA);
                auto q_cuda = q.to(torch::kCUDA);
                auto result_cuda = torch::special::zeta(x_cuda, q_cuda);
                auto result_cpu = result_cuda.to(torch::kCPU);
                (void)result_cpu;
            } catch (const c10::Error &e) {
                // CUDA operations might fail
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}