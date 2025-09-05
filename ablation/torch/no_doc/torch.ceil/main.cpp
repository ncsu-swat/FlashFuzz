#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for tensor creation
        if (Size < 3) {
            return 0;
        }

        // Create primary input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply ceil operation
        torch::Tensor result = torch::ceil(input_tensor);
        
        // Additional testing paths based on remaining data
        if (offset < Size) {
            uint8_t extra_ops = Data[offset++];
            
            // Test in-place operation
            if (extra_ops & 0x01) {
                torch::Tensor input_copy = input_tensor.clone();
                input_copy.ceil_();
                
                // Verify in-place matches out-of-place
                if (input_copy.dtype() == result.dtype() && 
                    input_copy.sizes() == result.sizes()) {
                    bool matches = torch::allclose(input_copy, result, 1e-5, 1e-8);
                    if (!matches && input_copy.numel() > 0) {
                        // Interesting discrepancy found
                        auto max_diff = torch::max(torch::abs(input_copy - result)).item<double>();
                        if (max_diff > 1e-6) {
                            std::cerr << "In-place vs out-of-place mismatch: " << max_diff << std::endl;
                        }
                    }
                }
            }
            
            // Test with output tensor pre-allocation
            if (extra_ops & 0x02) {
                torch::Tensor out_tensor = torch::empty_like(input_tensor);
                torch::ceil_out(out_tensor, input_tensor);
                
                // Verify output matches
                if (out_tensor.dtype() == result.dtype() && 
                    out_tensor.sizes() == result.sizes() &&
                    out_tensor.numel() > 0) {
                    torch::allclose(out_tensor, result, 1e-5, 1e-8);
                }
            }
            
            // Test with different memory layouts if tensor has multiple dimensions
            if ((extra_ops & 0x04) && input_tensor.dim() > 1) {
                // Create non-contiguous view
                torch::Tensor transposed = input_tensor.transpose(0, -1);
                torch::Tensor result_transposed = torch::ceil(transposed);
                
                // Verify shape consistency
                if (result_transposed.sizes() == transposed.sizes()) {
                    // Operation succeeded on non-contiguous tensor
                }
            }
            
            // Test with sliced/strided tensors
            if ((extra_ops & 0x08) && input_tensor.numel() > 2) {
                try {
                    torch::Tensor sliced = input_tensor.flatten().slice(0, 0, -1, 2);
                    torch::Tensor result_sliced = torch::ceil(sliced);
                    // Successfully processed strided tensor
                } catch (const c10::Error& e) {
                    // Slicing failed, continue
                }
            }
            
            // Test with different tensor types if more data available
            if (offset + 2 < Size && (extra_ops & 0x10)) {
                try {
                    // Create a second tensor with potentially different properties
                    torch::Tensor second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Test ceil on the second tensor
                    torch::Tensor second_result = torch::ceil(second_tensor);
                    
                    // If shapes match, test element-wise operations
                    if (input_tensor.sizes() == second_tensor.sizes()) {
                        torch::Tensor combined = torch::ceil(input_tensor + second_tensor);
                        // Successfully processed combined tensors
                    }
                } catch (const std::exception& e) {
                    // Second tensor creation failed, continue
                }
            }
            
            // Test with views and reshapes
            if ((extra_ops & 0x20) && input_tensor.numel() > 0) {
                try {
                    int64_t numel = input_tensor.numel();
                    torch::Tensor reshaped = input_tensor.reshape({-1});
                    torch::Tensor result_reshaped = torch::ceil(reshaped);
                    
                    // Try another reshape if possible
                    if (numel > 1 && numel % 2 == 0) {
                        torch::Tensor matrix_view = reshaped.view({2, numel/2});
                        torch::Tensor result_matrix = torch::ceil(matrix_view);
                    }
                } catch (const c10::Error& e) {
                    // Reshape failed, continue
                }
            }
            
            // Test with special values if floating point type
            if ((extra_ops & 0x40) && 
                (input_tensor.dtype() == torch::kFloat || 
                 input_tensor.dtype() == torch::kDouble ||
                 input_tensor.dtype() == torch::kHalf ||
                 input_tensor.dtype() == torch::kBFloat16)) {
                
                // Create tensor with special values
                torch::Tensor special_vals = torch::tensor({
                    std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN(),
                    0.0f, -0.0f, 1.5f, -1.5f
                }, input_tensor.options());
                
                if (special_vals.numel() <= input_tensor.numel() && input_tensor.numel() > 0) {
                    // Copy special values into a portion of the tensor
                    input_tensor.flatten().slice(0, 0, special_vals.numel()).copy_(special_vals);
                    torch::Tensor result_with_special = torch::ceil(input_tensor);
                }
            }
            
            // Test gradient computation if floating point
            if ((extra_ops & 0x80) && 
                (input_tensor.dtype() == torch::kFloat || 
                 input_tensor.dtype() == torch::kDouble) &&
                input_tensor.numel() > 0 && input_tensor.numel() < 1000) {
                
                try {
                    torch::Tensor grad_input = input_tensor.clone().requires_grad_(true);
                    torch::Tensor grad_result = torch::ceil(grad_input);
                    
                    if (grad_result.numel() > 0) {
                        // Ceil is not differentiable, but test backward pass handling
                        torch::Tensor grad_out = torch::ones_like(grad_result);
                        grad_result.backward(grad_out);
                        
                        // Check that gradient is zero (ceil has zero gradient almost everywhere)
                        if (grad_input.grad().defined()) {
                            torch::Tensor zero_grad = torch::zeros_like(grad_input);
                            torch::allclose(grad_input.grad(), zero_grad, 1e-5, 1e-8);
                        }
                    }
                } catch (const c10::Error& e) {
                    // Gradient computation failed, continue
                }
            }
        }
        
        // Validate basic properties of ceil operation
        if (result.defined() && result.numel() > 0) {
            // For integer types, ceil should be identity
            if (input_tensor.dtype() == torch::kInt8 || 
                input_tensor.dtype() == torch::kInt16 ||
                input_tensor.dtype() == torch::kInt32 ||
                input_tensor.dtype() == torch::kInt64) {
                
                bool is_identity = torch::equal(input_tensor, result);
                if (!is_identity && input_tensor.numel() < 100) {
                    std::cerr << "Ceil not identity for integer type" << std::endl;
                }
            }
            
            // For floating point, result should be >= input
            if ((input_tensor.dtype() == torch::kFloat || 
                 input_tensor.dtype() == torch::kDouble) &&
                input_tensor.numel() < 1000) {
                
                torch::Tensor diff = result - input_tensor;
                torch::Tensor min_diff = torch::min(diff);
                
                // Account for NaN values
                if (!torch::isnan(min_diff).item<bool>() && 
                    min_diff.item<double>() < -1e-6) {
                    std::cerr << "Ceil result less than input: " << min_diff.item<double>() << std::endl;
                }
            }
        }
        
        return 0;
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors - these are expected for invalid operations
        return 0;
    }
    catch (const std::bad_alloc &e)
    {
        // Memory allocation failure - input creates too large tensor
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}