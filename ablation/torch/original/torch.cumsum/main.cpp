#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Minimum size check: need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;  // Not enough data to create meaningful test
        }

        // Create input tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // If no data left for dimension, use a simple default
        if (offset >= Size) {
            // Test with dimension 0
            if (input.dim() > 0) {
                torch::Tensor result = torch::cumsum(input, 0);
            }
            return 0;
        }

        // Parse dimension for cumsum operation
        int64_t dim = 0;
        if (offset < Size) {
            uint8_t dim_byte = Data[offset++];
            
            // Handle both positive and negative dimensions
            if (input.dim() > 0) {
                // Map byte to valid dimension range [-input.dim(), input.dim()-1]
                int64_t total_range = 2 * input.dim();
                dim = (dim_byte % total_range) - input.dim();
            }
        }

        // Parse optional dtype for output
        torch::optional<torch::ScalarType> output_dtype = torch::nullopt;
        if (offset < Size) {
            uint8_t dtype_flag = Data[offset++];
            if (dtype_flag % 2 == 0 && offset < Size) {
                // Use a specific dtype for output
                uint8_t dtype_selector = Data[offset++];
                output_dtype = fuzzer_utils::parseDataType(dtype_selector);
            }
        }

        // Decide whether to use pre-allocated output tensor
        torch::Tensor output;
        bool use_out_tensor = false;
        if (offset < Size) {
            uint8_t out_flag = Data[offset++];
            if (out_flag % 3 == 0) {  // 33% chance to use output tensor
                use_out_tensor = true;
                
                // Create output tensor with same shape as input
                if (output_dtype.has_value()) {
                    output = torch::empty(input.sizes(), torch::TensorOptions().dtype(output_dtype.value()));
                } else {
                    output = torch::empty_like(input);
                }
            }
        }

        // Test various tensor configurations
        if (offset < Size) {
            uint8_t config_byte = Data[offset++];
            
            // Make tensor non-contiguous sometimes
            if (config_byte % 4 == 0 && input.dim() >= 2) {
                input = input.transpose(0, 1);
            }
            
            // Test with views/slices
            if (config_byte % 5 == 1 && input.numel() > 1) {
                auto sizes = input.sizes().vec();
                for (auto& s : sizes) {
                    if (s > 1) {
                        s = s / 2;
                        break;
                    }
                }
                if (sizes != input.sizes().vec()) {
                    input = input.view(sizes);
                }
            }
        }

        // Main operation: torch.cumsum with various configurations
        torch::Tensor result;
        
        // Test different API variations
        if (use_out_tensor) {
            // Use pre-allocated output tensor
            if (output_dtype.has_value()) {
                // This tests dtype conversion with output tensor
                torch::cumsum_out(output, input.to(output_dtype.value()), dim);
                result = output;
            } else {
                torch::cumsum_out(output, input, dim);
                result = output;
            }
        } else if (output_dtype.has_value()) {
            // Test with dtype conversion
            result = torch::cumsum(input, dim, output_dtype.value());
        } else {
            // Basic cumsum
            result = torch::cumsum(input, dim);
        }

        // Additional edge case testing based on remaining data
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            // Test cumsum on result (double cumsum)
            if (edge_case % 7 == 0 && result.dim() > 0) {
                int64_t dim2 = (edge_case / 7) % result.dim();
                torch::Tensor double_cumsum = torch::cumsum(result, dim2);
            }
            
            // Test with zero-dim tensor
            if (edge_case % 11 == 1) {
                torch::Tensor scalar = torch::tensor(3.14);
                torch::Tensor scalar_result = torch::cumsum(scalar, 0);
            }
            
            // Test with empty tensor
            if (edge_case % 13 == 2) {
                torch::Tensor empty = torch::empty({0, 3, 4});
                if (empty.dim() > 0) {
                    torch::Tensor empty_result = torch::cumsum(empty, 0);
                }
            }
            
            // Test extreme dimensions
            if (edge_case % 17 == 3) {
                // Test with very large dimension index (should throw)
                try {
                    torch::Tensor bad_dim_result = torch::cumsum(input, 1000);
                } catch (const c10::Error& e) {
                    // Expected for out-of-bounds dimension
                }
                
                // Test with very negative dimension index
                try {
                    torch::Tensor bad_neg_dim_result = torch::cumsum(input, -1000);
                } catch (const c10::Error& e) {
                    // Expected for out-of-bounds dimension
                }
            }
            
            // Test inplace operations on result
            if (edge_case % 19 == 4 && result.numel() > 0) {
                result.add_(1.0);  // Modify result to ensure it's writable
            }
        }

        // Test special floating point values if applicable
        if (input.is_floating_point() && offset < Size) {
            uint8_t special_val = Data[offset++];
            
            // Create tensors with special values
            if (special_val % 5 == 0) {
                torch::Tensor inf_tensor = torch::full_like(input, std::numeric_limits<float>::infinity());
                torch::Tensor inf_result = torch::cumsum(inf_tensor, dim);
            }
            
            if (special_val % 5 == 1) {
                torch::Tensor nan_tensor = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
                torch::Tensor nan_result = torch::cumsum(nan_tensor, dim);
            }
            
            if (special_val % 5 == 2) {
                torch::Tensor neg_inf_tensor = torch::full_like(input, -std::numeric_limits<float>::infinity());
                torch::Tensor neg_inf_result = torch::cumsum(neg_inf_tensor, dim);
            }
        }

        // Test with requires_grad if floating point
        if (input.is_floating_point() && offset < Size) {
            uint8_t grad_flag = Data[offset++];
            if (grad_flag % 3 == 0) {
                input.requires_grad_(true);
                torch::Tensor grad_result = torch::cumsum(input, dim);
                
                // Test backward pass
                if (grad_result.numel() > 0) {
                    torch::Tensor grad_out = torch::ones_like(grad_result);
                    grad_result.backward(grad_out);
                }
            }
        }

        // Validate result properties
        if (result.defined()) {
            // Check shape preservation
            if (result.sizes() != input.sizes()) {
                std::cerr << "Warning: cumsum changed tensor shape!" << std::endl;
            }
            
            // For integer types, verify no overflow occurred silently
            if (input.is_integral() && !input.dtype().is_bool()) {
                // This is just a sanity check, not a strict validation
                auto input_sum = input.sum();
                auto result_last = result.select(dim, result.size(dim) - 1).sum();
                // Note: This comparison might not be exact due to different summation orders
            }
        }

    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors (like dimension out of bounds) are expected
        // Don't print these as they're part of normal fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    catch (...)
    {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0; // keep the input
}