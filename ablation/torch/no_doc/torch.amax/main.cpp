#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for tensor creation and operation parameters
        if (Size < 4) {
            return 0;
        }

        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // If we've consumed all data, still try the operation with default parameters
        if (offset >= Size) {
            try {
                // Test with no dimension specified (reduces over all dimensions)
                torch::Tensor result = torch::amax(input);
                return 0;
            } catch (const c10::Error& e) {
                // PyTorch internal errors are expected for invalid operations
                return 0;
            }
        }

        // Parse keepdim flag
        bool keepdim = (offset < Size) ? (Data[offset++] & 1) : false;
        
        // Parse whether to specify dimensions
        bool use_dims = (offset < Size) ? (Data[offset++] & 1) : true;
        
        if (!use_dims) {
            // Test without specifying dimensions (reduces over all)
            try {
                torch::Tensor result = torch::amax(input);
            } catch (const c10::Error& e) {
                // Expected for some edge cases
            }
            return 0;
        }
        
        // Parse number of dimensions to reduce over
        uint8_t num_reduce_dims = 0;
        if (offset < Size) {
            // Limit to actual tensor rank to avoid excessive memory allocation
            num_reduce_dims = Data[offset++] % (input.dim() + 1);
        }
        
        // Parse dimension indices
        std::vector<int64_t> dims;
        if (num_reduce_dims > 0 && input.dim() > 0) {
            for (uint8_t i = 0; i < num_reduce_dims && offset < Size; ++i) {
                // Allow both positive and negative indexing
                int8_t dim_byte = static_cast<int8_t>(Data[offset++]);
                int64_t dim;
                
                if (dim_byte < 0) {
                    // Negative indexing from the end
                    dim = dim_byte;  // Will be in range [-128, -1]
                } else {
                    // Positive indexing from the start
                    dim = dim_byte % input.dim();
                }
                
                dims.push_back(dim);
            }
        }
        
        // Test various amax operations
        try {
            if (dims.empty()) {
                // Test with no dimensions specified
                torch::Tensor result1 = torch::amax(input);
                
                // Also test the variant that returns values and indices
                if (input.numel() > 0) {
                    // Test on flattened tensor for max with indices
                    torch::Tensor flat = input.flatten();
                    auto [values, indices] = torch::max(flat);
                }
            } else if (dims.size() == 1) {
                // Single dimension reduction
                torch::Tensor result2 = torch::amax(input, dims[0], keepdim);
                
                // Also test torch::max variant which returns both values and indices
                if (input.dim() > 0 && dims[0] >= -input.dim() && dims[0] < input.dim()) {
                    auto [max_vals, max_indices] = torch::max(input, dims[0], keepdim);
                }
            } else {
                // Multiple dimension reduction
                torch::Tensor result3 = torch::amax(input, dims, keepdim);
            }
            
            // Test edge cases with special values if tensor has elements
            if (input.numel() > 0 && input.is_floating_point()) {
                // Create tensor with NaN/Inf values to test handling
                torch::Tensor special_tensor = input.clone();
                if (special_tensor.numel() >= 3) {
                    special_tensor.view(-1)[0] = std::numeric_limits<float>::quiet_NaN();
                    special_tensor.view(-1)[1] = std::numeric_limits<float>::infinity();
                    special_tensor.view(-1)[2] = -std::numeric_limits<float>::infinity();
                }
                
                try {
                    torch::Tensor result_special = torch::amax(special_tensor);
                } catch (const c10::Error& e) {
                    // Some operations might fail with special values
                }
            }
            
            // Test with different memory layouts if tensor is multi-dimensional
            if (input.dim() >= 2) {
                // Test with transposed tensor (non-contiguous)
                torch::Tensor transposed = input.transpose(0, -1);
                try {
                    torch::Tensor result_t = torch::amax(transposed);
                } catch (const c10::Error& e) {
                    // Expected for some cases
                }
                
                // Test with permuted tensor
                std::vector<int64_t> perm_dims;
                for (int64_t i = input.dim() - 1; i >= 0; --i) {
                    perm_dims.push_back(i);
                }
                torch::Tensor permuted = input.permute(perm_dims);
                try {
                    torch::Tensor result_p = torch::amax(permuted);
                } catch (const c10::Error& e) {
                    // Expected for some cases
                }
            }
            
            // Test with views and slices if tensor is large enough
            if (input.numel() > 1) {
                // Test with a view
                torch::Tensor viewed = input.view({-1});
                try {
                    torch::Tensor result_v = torch::amax(viewed);
                } catch (const c10::Error& e) {
                    // Expected for some cases
                }
                
                // Test with a slice if multi-dimensional
                if (input.dim() > 0 && input.size(0) > 1) {
                    torch::Tensor sliced = input.narrow(0, 0, 1);
                    try {
                        torch::Tensor result_s = torch::amax(sliced);
                    } catch (const c10::Error& e) {
                        // Expected for some cases
                    }
                }
            }
            
            // Test empty tensor edge cases
            if (input.numel() == 0) {
                try {
                    // This might throw for empty tensors
                    torch::Tensor empty_result = torch::amax(input);
                } catch (const c10::Error& e) {
                    // Expected behavior for empty tensors
                }
                
                if (!dims.empty()) {
                    try {
                        torch::Tensor empty_result_dims = torch::amax(input, dims, keepdim);
                    } catch (const c10::Error& e) {
                        // Expected behavior
                    }
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch errors are expected for invalid operations
            // Continue fuzzing
        } catch (const std::runtime_error& e) {
            // Runtime errors from invalid dimensions etc.
            // Continue fuzzing
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}