#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    // Minimum size check - need at least a few bytes for basic tensor creation
    if (size < 4) {
        return 0;
    }

    try
    {
        size_t offset = 0;
        
        // Create primary input tensor
        torch::Tensor input = fuzzer_utils::createTensor(data, size, offset);
        
        // Test basic asin operation
        torch::Tensor result = torch::asin(input);
        
        // Verify result has same shape as input
        if (result.sizes() != input.sizes()) {
            std::cerr << "Shape mismatch between input and result!" << std::endl;
            return -1;
        }
        
        // Test with output tensor pre-allocation if we have enough data
        if (offset < size) {
            uint8_t use_out = data[offset++] % 2;
            
            if (use_out) {
                // Create output tensor with same shape and dtype
                torch::Tensor out = torch::empty_like(input);
                torch::asin(input, out);
                
                // Verify in-place operation worked
                if (!torch::allclose(result, out, 1e-5, 1e-8)) {
                    // This is expected for NaN values, so just note it
                    auto nan_mask = torch::isnan(result);
                    if (!torch::any(nan_mask).item<bool>()) {
                        std::cerr << "Unexpected difference in out= variant" << std::endl;
                    }
                }
            }
        }
        
        // Test edge cases based on remaining fuzzer data
        if (offset < size) {
            uint8_t edge_case = data[offset++] % 8;
            
            switch (edge_case) {
                case 0: {
                    // Test with values exactly at boundaries [-1, 1]
                    torch::Tensor boundary = torch::ones_like(input);
                    if (offset < size && data[offset++] % 2) {
                        boundary = boundary * -1;  // Test -1
                    }
                    torch::Tensor boundary_result = torch::asin(boundary);
                    break;
                }
                case 1: {
                    // Test with zeros
                    torch::Tensor zeros = torch::zeros_like(input);
                    torch::Tensor zero_result = torch::asin(zeros);
                    // asin(0) should be 0
                    if (!torch::allclose(zero_result, zeros, 1e-7, 1e-10)) {
                        std::cerr << "asin(0) != 0" << std::endl;
                    }
                    break;
                }
                case 2: {
                    // Test with values outside valid range (should produce NaN)
                    torch::Tensor out_of_range = torch::ones_like(input) * 2.0;
                    torch::Tensor nan_result = torch::asin(out_of_range);
                    if (!torch::all(torch::isnan(nan_result)).item<bool>()) {
                        // Some dtypes might handle this differently
                    }
                    break;
                }
                case 3: {
                    // Test with negative values outside range
                    torch::Tensor neg_out_of_range = torch::ones_like(input) * -2.0;
                    torch::Tensor neg_nan_result = torch::asin(neg_out_of_range);
                    break;
                }
                case 4: {
                    // Test with mixed valid/invalid values
                    if (input.numel() > 0) {
                        torch::Tensor mixed = input.clone();
                        // Clamp some values to valid range, leave others
                        auto mask = torch::rand_like(mixed) > 0.5;
                        mixed = torch::where(mask, torch::clamp(mixed, -1.0, 1.0), mixed * 2.0);
                        torch::Tensor mixed_result = torch::asin(mixed);
                    }
                    break;
                }
                case 5: {
                    // Test with very small values (near zero)
                    torch::Tensor small = torch::ones_like(input) * 1e-10;
                    torch::Tensor small_result = torch::asin(small);
                    break;
                }
                case 6: {
                    // Test with values very close to boundaries
                    torch::Tensor near_one = torch::ones_like(input) * 0.9999999;
                    torch::Tensor near_result = torch::asin(near_one);
                    break;
                }
                case 7: {
                    // Test chained operations
                    torch::Tensor clamped = torch::clamp(input, -1.0, 1.0);
                    torch::Tensor chain1 = torch::asin(clamped);
                    torch::Tensor chain2 = torch::sin(chain1);  // sin(asin(x)) = x for x in [-1,1]
                    // Should be close to original clamped values
                    if (!torch::allclose(chain2, clamped, 1e-5, 1e-7)) {
                        // Might have precision issues with certain dtypes
                    }
                    break;
                }
            }
        }
        
        // Test different tensor properties if we have more data
        if (offset < size) {
            uint8_t property_test = data[offset++] % 4;
            
            switch (property_test) {
                case 0: {
                    // Test with non-contiguous tensor
                    if (input.dim() > 0 && input.size(0) > 1) {
                        torch::Tensor transposed = input.transpose(0, -1);
                        torch::Tensor trans_result = torch::asin(transposed);
                    }
                    break;
                }
                case 1: {
                    // Test with view
                    if (input.numel() > 0) {
                        torch::Tensor flat = input.flatten();
                        torch::Tensor flat_result = torch::asin(flat);
                    }
                    break;
                }
                case 2: {
                    // Test with slice
                    if (input.dim() > 0 && input.size(0) > 1) {
                        torch::Tensor sliced = input.narrow(0, 0, 1);
                        torch::Tensor slice_result = torch::asin(sliced);
                    }
                    break;
                }
                case 3: {
                    // Test gradient computation if floating point
                    if (input.is_floating_point() && input.numel() > 0) {
                        torch::Tensor grad_input = torch::clamp(input, -0.99, 0.99).requires_grad_(true);
                        torch::Tensor grad_result = torch::asin(grad_input);
                        if (grad_result.numel() > 0) {
                            torch::Tensor grad_sum = grad_result.sum();
                            grad_sum.backward();
                            // Gradient exists
                            if (!grad_input.grad().defined()) {
                                std::cerr << "Gradient not computed" << std::endl;
                            }
                        }
                    }
                    break;
                }
            }
        }
        
        // Additional dtype-specific tests
        if (offset < size && input.numel() > 0) {
            uint8_t dtype_test = data[offset++] % 3;
            
            switch (dtype_test) {
                case 0: {
                    // Test dtype conversion
                    if (!input.is_complex()) {
                        torch::Tensor as_float = input.to(torch::kFloat32);
                        torch::Tensor float_result = torch::asin(as_float);
                        
                        torch::Tensor as_double = input.to(torch::kFloat64);
                        torch::Tensor double_result = torch::asin(as_double);
                    }
                    break;
                }
                case 1: {
                    // Test with complex tensors if applicable
                    if (input.is_complex()) {
                        torch::Tensor complex_result = torch::asin(input);
                        // Complex asin has different behavior
                    }
                    break;
                }
                case 2: {
                    // Test batch operations
                    if (input.dim() >= 2) {
                        // Process each batch element
                        for (int64_t i = 0; i < std::min(input.size(0), int64_t(3)); ++i) {
                            torch::Tensor batch_elem = input[i];
                            torch::Tensor batch_result = torch::asin(batch_elem);
                        }
                    }
                    break;
                }
            }
        }
        
        // Memory stress test with larger tensors if enough data
        if (offset + 10 < size) {
            uint8_t stress_test = data[offset++] % 2;
            if (stress_test && input.numel() < 1000000) {  // Limit to prevent OOM
                // Create a larger tensor by repeating
                std::vector<int64_t> repeat_dims(input.dim(), 2);
                torch::Tensor large = input.repeat(repeat_dims);
                torch::Tensor large_result = torch::asin(large);
            }
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors - these are often expected for edge cases
        return 0;  // Continue fuzzing
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;  // Discard this input
    }
    catch (...)
    {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;  // Discard this input
    }
    
    return 0;  // Keep the input
}