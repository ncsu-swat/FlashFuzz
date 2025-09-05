#include "fuzzer_utils.h"
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 2 bytes: 1 for number of tensors, 1 for axis
        if (Size < 2) {
            return 0;
        }
        
        // Parse number of tensors to concatenate (1-10)
        uint8_t num_tensors_raw = Data[offset++];
        size_t num_tensors = (num_tensors_raw % 10) + 1;
        
        // Parse axis for concatenation
        int8_t axis_raw;
        if (offset < Size) {
            axis_raw = static_cast<int8_t>(Data[offset++]);
        } else {
            axis_raw = 0;
        }
        
        // Create tensors from fuzzer input
        std::vector<torch::Tensor> tensors;
        tensors.reserve(num_tensors);
        
        for (size_t i = 0; i < num_tensors; ++i) {
            if (offset >= Size) {
                // If we run out of data, create a small random tensor
                auto dtype = fuzzer_utils::parseDataType(static_cast<uint8_t>(i));
                tensors.push_back(torch::randn({2, 3}, torch::TensorOptions().dtype(dtype)));
            } else {
                try {
                    torch::Tensor t = fuzzer_utils::createTensor(Data, Size, offset);
                    tensors.push_back(t);
                } catch (const std::exception& e) {
                    // If tensor creation fails, create a fallback tensor
                    tensors.push_back(torch::randn({2, 3}));
                }
            }
        }
        
        // Ensure we have at least one tensor
        if (tensors.empty()) {
            tensors.push_back(torch::randn({2, 3}));
        }
        
        // Determine valid axis range based on first tensor
        int64_t ndim = tensors[0].dim();
        int64_t axis = 0;
        
        if (ndim > 0) {
            // Map axis_raw to valid range [-ndim, ndim-1]
            axis = axis_raw % (2 * ndim);
            if (axis >= ndim) {
                axis = axis - 2 * ndim;
            }
        }
        
        // Test torch.cat (torch.concatenate is an alias)
        try {
            torch::Tensor result = torch::cat(tensors, axis);
            
            // Additional operations to increase coverage
            if (result.numel() > 0) {
                // Test various tensor properties
                bool is_contiguous = result.is_contiguous();
                auto dtype = result.dtype();
                auto device = result.device();
                auto sizes = result.sizes();
                
                // Test with different memory layouts if tensor is large enough
                if (result.numel() > 1 && result.dim() > 1) {
                    torch::Tensor transposed = result.transpose(0, -1);
                    torch::Tensor reshaped = result.reshape({-1});
                    
                    // Test concatenating the result with itself
                    if (result.dim() > 0) {
                        std::vector<torch::Tensor> double_tensors = {result, result};
                        torch::Tensor double_result = torch::cat(double_tensors, 0);
                    }
                }
                
                // Test edge cases with empty tensors
                if (offset < Size && Data[offset % Size] % 4 == 0) {
                    std::vector<torch::Tensor> mixed_tensors;
                    mixed_tensors.push_back(result);
                    mixed_tensors.push_back(torch::empty({0}));
                    try {
                        torch::Tensor mixed_result = torch::cat(mixed_tensors, 0);
                    } catch (...) {
                        // Ignore failures with empty tensor concatenation
                    }
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid operations
            // Continue fuzzing
        } catch (const std::exception& e) {
            // Log but continue for other exceptions
            std::cout << "Exception during concatenation: " << e.what() << std::endl;
        }
        
        // Test special cases based on remaining fuzzer input
        if (offset < Size) {
            uint8_t special_case = Data[offset++];
            
            // Test concatenating tensors with mixed dtypes
            if (special_case % 5 == 0 && tensors.size() >= 2) {
                try {
                    // Convert first tensor to different dtype
                    tensors[0] = tensors[0].to(torch::kFloat32);
                    tensors[1] = tensors[1].to(torch::kInt64);
                    torch::Tensor mixed_dtype_result = torch::cat(tensors, 0);
                } catch (...) {
                    // Expected to fail with mixed dtypes
                }
            }
            
            // Test concatenating along invalid axis
            if (special_case % 7 == 0) {
                try {
                    torch::Tensor invalid_axis_result = torch::cat(tensors, 100);
                } catch (...) {
                    // Expected to fail with invalid axis
                }
            }
            
            // Test with non-contiguous tensors
            if (special_case % 3 == 0 && tensors[0].dim() >= 2) {
                try {
                    std::vector<torch::Tensor> non_contiguous_tensors;
                    for (auto& t : tensors) {
                        if (t.dim() >= 2) {
                            non_contiguous_tensors.push_back(t.transpose(0, -1));
                        } else {
                            non_contiguous_tensors.push_back(t);
                        }
                    }
                    torch::Tensor non_contiguous_result = torch::cat(non_contiguous_tensors, 0);
                } catch (...) {
                    // Handle failures with non-contiguous tensors
                }
            }
            
            // Test concatenating single tensor (edge case)
            if (special_case % 11 == 0) {
                std::vector<torch::Tensor> single = {tensors[0]};
                try {
                    torch::Tensor single_result = torch::cat(single, 0);
                } catch (...) {
                    // Handle single tensor concatenation failures
                }
            }
        }
        
        // Test concatenating tensors with requires_grad
        if (offset < Size && Data[offset % Size] % 2 == 0) {
            try {
                for (auto& t : tensors) {
                    if (t.dtype() == torch::kFloat || t.dtype() == torch::kDouble || 
                        t.dtype() == torch::kFloat16 || t.dtype() == torch::kBFloat16) {
                        t = t.requires_grad_(true);
                    }
                }
                torch::Tensor grad_result = torch::cat(tensors, 0);
                
                // Test backward pass if applicable
                if (grad_result.requires_grad() && grad_result.numel() > 0) {
                    torch::Tensor grad_output = torch::ones_like(grad_result);
                    grad_result.backward(grad_output);
                }
            } catch (...) {
                // Ignore gradient-related failures
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}