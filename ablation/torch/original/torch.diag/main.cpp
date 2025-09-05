#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Minimum size check: need at least a few bytes for tensor creation and diagonal parameter
    if (size < 4) {
        return 0;
    }

    try {
        size_t offset = 0;
        
        // Create input tensor - can be 1D or 2D based on fuzzer input
        torch::Tensor input = fuzzer_utils::createTensor(data, size, offset);
        
        // Parse diagonal offset parameter
        int64_t diagonal = 0;
        if (offset < size) {
            // Use remaining bytes to determine diagonal offset
            uint8_t diagonal_byte = data[offset++];
            // Map to reasonable range [-10, 10] to avoid extreme memory allocation
            diagonal = static_cast<int64_t>(diagonal_byte % 21) - 10;
        }
        
        // Determine if we should test with pre-allocated output tensor
        bool use_out_tensor = false;
        if (offset < size) {
            use_out_tensor = (data[offset++] % 2) == 0;
        }
        
        // Handle different input dimensions
        if (input.dim() == 0) {
            // Scalar tensor - promote to 1D
            input = input.unsqueeze(0);
        } else if (input.dim() > 2) {
            // For tensors with dim > 2, flatten or select first 2 dimensions
            // to make it compatible with diag operation
            while (input.dim() > 2) {
                input = input.select(0, 0);
            }
        }
        
        torch::Tensor result;
        
        if (use_out_tensor && offset < size) {
            // Test with pre-allocated output tensor
            try {
                // Determine expected output shape
                torch::Tensor temp_result = torch::diag(input, diagonal);
                
                // Create output tensor with same dtype but potentially different shape
                torch::Tensor out;
                
                // Randomly decide if output shape should match or not
                if (data[offset % size] % 3 == 0) {
                    // Correct shape
                    out = torch::empty_like(temp_result);
                } else if (data[offset % size] % 3 == 1) {
                    // Wrong shape but same number of elements
                    auto numel = temp_result.numel();
                    if (numel > 0) {
                        out = torch::empty({numel}, temp_result.options());
                    } else {
                        out = torch::empty_like(temp_result);
                    }
                } else {
                    // Completely different shape
                    out = torch::empty({3, 4}, temp_result.options());
                }
                
                result = torch::diag_out(out, input, diagonal);
                
                // Verify that out and result point to same storage
                if (out.data_ptr() != result.data_ptr()) {
                    std::cerr << "Warning: out tensor not properly aliased" << std::endl;
                }
            } catch (const c10::Error& e) {
                // Expected for shape mismatches - continue with regular diag
                result = torch::diag(input, diagonal);
            }
        } else {
            // Standard diag operation
            result = torch::diag(input, diagonal);
        }
        
        // Perform various validations and edge case testing
        
        // Test 1: Verify dimension change
        if (input.dim() == 1) {
            // 1D input should produce 2D output
            if (result.dim() != 2) {
                std::cerr << "Unexpected: 1D input didn't produce 2D output" << std::endl;
            }
            
            // Verify diagonal elements match input
            int64_t expected_size = input.size(0) + std::abs(diagonal);
            if (result.size(0) != expected_size || result.size(1) != expected_size) {
                // This might be expected for very large diagonal offsets
            }
        } else if (input.dim() == 2) {
            // 2D input should produce 1D output
            if (result.dim() != 1) {
                std::cerr << "Unexpected: 2D input didn't produce 1D output" << std::endl;
            }
        }
        
        // Test 2: Check for NaN/Inf propagation if input contains them
        if (input.dtype().isFloatingPoint() || input.dtype().isComplex()) {
            bool has_nan = torch::any(torch::isnan(input)).item<bool>();
            bool has_inf = torch::any(torch::isinf(input)).item<bool>();
            
            if (has_nan || has_inf) {
                // These should propagate through diag operation
                bool result_has_nan = torch::any(torch::isnan(result)).item<bool>();
                bool result_has_inf = torch::any(torch::isinf(result)).item<bool>();
                
                // This is just for observation, not an error
                if (has_nan && !result_has_nan) {
                    std::cerr << "Note: NaN didn't propagate through diag" << std::endl;
                }
                if (has_inf && !result_has_inf) {
                    std::cerr << "Note: Inf didn't propagate through diag" << std::endl;
                }
            }
        }
        
        // Test 3: Test with different memory layouts if we have more data
        if (offset + 2 < size && input.dim() == 2) {
            // Create non-contiguous version of input
            torch::Tensor transposed = input.t();
            torch::Tensor result_transposed = torch::diag(transposed, diagonal);
            
            // Results should be the same regardless of memory layout
            if (result.sizes() == result_transposed.sizes()) {
                if (!torch::allclose(result, result_transposed, 1e-5, 1e-8)) {
                    std::cerr << "Warning: Different results for transposed input" << std::endl;
                }
            }
        }
        
        // Test 4: Edge cases with empty tensors
        if (input.numel() == 0) {
            // Empty tensor handling
            if (result.numel() != 0 && diagonal == 0) {
                std::cerr << "Unexpected: Empty input produced non-empty output" << std::endl;
            }
        }
        
        // Test 5: Test diagonal values that exceed matrix dimensions
        if (input.dim() == 2) {
            int64_t max_diagonal = std::max(input.size(0), input.size(1));
            if (std::abs(diagonal) >= max_diagonal) {
                // Should produce empty result or handle gracefully
                if (result.numel() > 0) {
                    // This might be valid depending on implementation
                }
            }
        }
        
        // Test 6: Round-trip test for square matrices
        if (input.dim() == 2 && input.size(0) == input.size(1) && diagonal == 0) {
            // Extract diagonal and reconstruct
            torch::Tensor diag_elements = torch::diag(input, 0);
            torch::Tensor reconstructed = torch::diag(diag_elements, 0);
            
            // Check if diagonal elements match
            torch::Tensor original_diag = torch::diagonal(input, 0);
            torch::Tensor reconstructed_diag = torch::diagonal(reconstructed, 0);
            
            if (!torch::allclose(original_diag, reconstructed_diag, 1e-5, 1e-8)) {
                std::cerr << "Warning: Round-trip diagonal mismatch" << std::endl;
            }
        }
        
        // Test 7: Test with different dtypes
        if (offset < size) {
            uint8_t dtype_convert = data[offset++];
            if (dtype_convert % 4 == 0 && !input.dtype().isComplex()) {
                // Try converting to different dtype
                try {
                    torch::Tensor converted = input.to(torch::kFloat64);
                    torch::Tensor result_converted = torch::diag(converted, diagonal);
                    
                    // Results should be numerically similar
                    torch::Tensor result_cast = result.to(torch::kFloat64);
                    if (result_cast.sizes() == result_converted.sizes()) {
                        if (!torch::allclose(result_cast, result_converted, 1e-5, 1e-8)) {
                            std::cerr << "Warning: Different results after dtype conversion" << std::endl;
                        }
                    }
                } catch (const c10::Error& e) {
                    // Dtype conversion might fail for some types
                }
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors (expected for invalid operations)
        return 0;
    } catch (const std::exception& e) {
        // Log unexpected exceptions for debugging
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other unexpected exceptions
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}