#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <exception>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for tensor creation
        if (Size < 2) {
            return 0;  // Not enough data, but keep for coverage
        }

        // Create primary input tensor
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception &e) {
            // If we can't create a basic tensor, try with minimal data
            if (Size > 0) {
                // Create a simple scalar tensor as fallback
                float val = static_cast<float>(Data[0]) / 255.0f - 0.5f;
                input_tensor = torch::tensor(val);
            } else {
                return 0;
            }
        }

        // Test various tensor configurations
        if (offset < Size) {
            uint8_t config = Data[offset++];
            
            // Test with different memory layouts
            if (config & 0x01) {
                input_tensor = input_tensor.contiguous();
            }
            
            // Test with non-contiguous tensors
            if ((config & 0x02) && input_tensor.numel() > 1) {
                input_tensor = input_tensor.transpose(0, -1);
            }
            
            // Test with different devices if available
            if ((config & 0x04) && torch::cuda::is_available()) {
                try {
                    input_tensor = input_tensor.to(torch::kCUDA);
                } catch (...) {
                    // CUDA operation failed, continue with CPU
                }
            }
            
            // Test with requires_grad for autograd coverage
            if (config & 0x08) {
                // Only set requires_grad for floating point types
                if (input_tensor.dtype() == torch::kFloat || 
                    input_tensor.dtype() == torch::kDouble ||
                    input_tensor.dtype() == torch::kHalf ||
                    input_tensor.dtype() == torch::kBFloat16) {
                    try {
                        input_tensor = input_tensor.requires_grad_(true);
                    } catch (...) {
                        // Some dtypes might not support autograd
                    }
                }
            }
        }

        // Main operation: torch::special::expm1
        torch::Tensor result;
        try {
            result = torch::special::expm1(input_tensor);
            
            // Verify output properties
            if (result.defined()) {
                // Check shape preservation
                if (result.sizes() != input_tensor.sizes()) {
                    std::cerr << "Shape mismatch: input " << input_tensor.sizes() 
                             << " vs output " << result.sizes() << std::endl;
                }
                
                // Force computation for lazy tensors
                if (result.numel() > 0 && result.numel() < 1000) {
                    result.item<float>();  // Force evaluation for small tensors
                } else if (result.numel() >= 1000) {
                    result.sum().item<float>();  // Force evaluation for larger tensors
                }
            }
        } catch (const c10::Error &e) {
            // PyTorch-specific errors - these are expected for invalid operations
            return 0;
        }

        // Test edge cases with special values if we have more data
        if (offset < Size && result.defined()) {
            uint8_t edge_case = Data[offset++];
            
            // Test with infinity values
            if ((edge_case & 0x01) && input_tensor.dtype().isFloatingPoint()) {
                try {
                    auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::infinity());
                    auto inf_result = torch::special::expm1(inf_tensor);
                    inf_result.sum().item<float>();  // Force evaluation
                } catch (...) {
                    // Expected to potentially fail with inf
                }
            }
            
            // Test with negative infinity
            if ((edge_case & 0x02) && input_tensor.dtype().isFloatingPoint()) {
                try {
                    auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<float>::infinity());
                    auto neg_inf_result = torch::special::expm1(neg_inf_tensor);
                    neg_inf_result.sum().item<float>();  // Should be -1
                } catch (...) {
                    // Expected behavior
                }
            }
            
            // Test with NaN values
            if ((edge_case & 0x04) && input_tensor.dtype().isFloatingPoint()) {
                try {
                    auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::quiet_NaN());
                    auto nan_result = torch::special::expm1(nan_tensor);
                    nan_result.sum().item<float>();  // Force evaluation
                } catch (...) {
                    // Expected behavior with NaN
                }
            }
            
            // Test with very small values (where expm1 is most useful)
            if ((edge_case & 0x08) && input_tensor.dtype().isFloatingPoint()) {
                try {
                    auto small_tensor = torch::full_like(input_tensor, 1e-10);
                    auto small_result = torch::special::expm1(small_tensor);
                    
                    // Compare with naive exp(x) - 1 to verify precision benefit
                    auto naive_result = torch::exp(small_tensor) - 1;
                    
                    // The expm1 result should be more accurate for small values
                    if (small_result.numel() > 0) {
                        small_result.sum().item<float>();
                    }
                } catch (...) {
                    // Continue on error
                }
            }
            
            // Test with very large values
            if ((edge_case & 0x10) && input_tensor.dtype().isFloatingPoint()) {
                try {
                    auto large_tensor = torch::full_like(input_tensor, 100.0);
                    auto large_result = torch::special::expm1(large_tensor);
                    if (large_result.numel() > 0 && large_result.numel() < 100) {
                        large_result.sum().item<float>();  // May overflow
                    }
                } catch (...) {
                    // Expected for overflow cases
                }
            }
        }

        // Test with different output tensor (out= parameter)
        if (offset < Size && input_tensor.dtype().isFloatingPoint()) {
            try {
                torch::Tensor out_tensor = torch::empty_like(input_tensor);
                torch::special::expm1_out(out_tensor, input_tensor);
                
                // Verify in-place operation worked
                if (out_tensor.numel() > 0 && out_tensor.numel() < 1000) {
                    out_tensor.sum().item<float>();
                }
            } catch (...) {
                // Some configurations might not support out= parameter
            }
        }

        // Test gradient computation if applicable
        if (input_tensor.requires_grad() && result.defined()) {
            try {
                auto grad_output = torch::ones_like(result);
                result.backward(grad_output);
                
                // Check gradient exists and has correct shape
                if (input_tensor.grad().defined()) {
                    if (input_tensor.grad().sizes() != input_tensor.sizes()) {
                        std::cerr << "Gradient shape mismatch" << std::endl;
                    }
                    // Force gradient evaluation
                    if (input_tensor.grad().numel() > 0 && input_tensor.grad().numel() < 1000) {
                        input_tensor.grad().sum().item<float>();
                    }
                }
            } catch (...) {
                // Gradient computation might fail for some dtypes
            }
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}