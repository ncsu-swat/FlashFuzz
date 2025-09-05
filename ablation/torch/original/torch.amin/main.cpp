#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper function to parse dimensions for reduction
std::vector<int64_t> parseDimensions(const uint8_t* data, size_t& offset, size_t size, int64_t tensor_rank) {
    std::vector<int64_t> dims;
    
    if (offset >= size) {
        return dims;
    }
    
    // Parse number of dimensions to reduce (0 means reduce all)
    uint8_t num_dims = data[offset++] % (tensor_rank + 1);
    
    if (num_dims == 0) {
        // Reduce all dimensions
        for (int64_t i = 0; i < tensor_rank; ++i) {
            dims.push_back(i);
        }
        return dims;
    }
    
    // Parse specific dimensions
    for (uint8_t i = 0; i < num_dims && offset < size; ++i) {
        // Map to valid dimension range [-tensor_rank, tensor_rank)
        int8_t dim_raw = static_cast<int8_t>(data[offset++]);
        int64_t dim = dim_raw % tensor_rank;
        if (dim < 0) {
            dim += tensor_rank; // Handle negative indexing
        }
        
        // Avoid duplicate dimensions
        if (std::find(dims.begin(), dims.end(), dim) == dims.end()) {
            dims.push_back(dim);
        }
    }
    
    return dims;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        if (size < 3) {
            // Need at least: dtype(1) + rank(1) + keepdim(1)
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(data, size, offset);
        
        if (offset >= size) {
            // Not enough data for operation parameters
            // Still try with default parameters
            torch::Tensor result = torch::amin(input);
            return 0;
        }
        
        // Parse keepdim parameter
        bool keepdim = (data[offset++] & 1) != 0;
        
        // Get tensor rank for dimension parsing
        int64_t tensor_rank = input.dim();
        
        if (tensor_rank == 0) {
            // Scalar tensor - amin should work but dimensions don't apply
            torch::Tensor result = torch::amin(input);
            return 0;
        }
        
        // Parse dimensions to reduce
        std::vector<int64_t> dims = parseDimensions(data, offset, size, tensor_rank);
        
        // Test different invocation patterns
        if (dims.empty()) {
            // Test 1: Reduce all dimensions (no dim specified)
            torch::Tensor result1 = torch::amin(input);
            
            // Test 2: Also try with explicit keepdim
            if (offset < size && (data[offset] & 1)) {
                torch::Tensor result2 = torch::amin(input, /*dim=*/{}, keepdim);
            }
        } else if (dims.size() == 1) {
            // Test single dimension reduction
            int64_t dim = dims[0];
            
            // Validate dimension is in range
            if (dim >= -tensor_rank && dim < tensor_rank) {
                torch::Tensor result = torch::amin(input, dim, keepdim);
                
                // Verify output shape
                if (keepdim) {
                    // Output should have same rank as input
                    if (result.dim() != input.dim()) {
                        std::cerr << "Unexpected rank change with keepdim=true" << std::endl;
                    }
                } else {
                    // Output should have rank reduced by 1
                    if (result.dim() != std::max(int64_t(0), input.dim() - 1)) {
                        std::cerr << "Unexpected rank with keepdim=false" << std::endl;
                    }
                }
                
                // Test with optional output tensor
                if (offset < size && (data[offset] & 1)) {
                    torch::Tensor out = torch::empty_like(result);
                    torch::amin_out(out, input, dim, keepdim);
                }
            }
        } else {
            // Test multiple dimension reduction
            torch::Tensor result = torch::amin(input, dims, keepdim);
            
            // Verify output shape
            if (keepdim) {
                // Output should have same rank as input
                if (result.dim() != input.dim()) {
                    std::cerr << "Unexpected rank change with keepdim=true (multi-dim)" << std::endl;
                }
            } else {
                // Output rank should be reduced by number of unique dimensions
                int64_t expected_rank = std::max(int64_t(0), input.dim() - static_cast<int64_t>(dims.size()));
                if (result.dim() != expected_rank) {
                    std::cerr << "Unexpected rank with keepdim=false (multi-dim)" << std::endl;
                }
            }
            
            // Test with optional output tensor
            if (offset < size && (data[offset] & 1)) {
                torch::Tensor out = torch::empty_like(result);
                torch::amin_out(out, input, dims, keepdim);
            }
        }
        
        // Additional edge case testing based on remaining data
        if (offset + 1 < size) {
            uint8_t edge_case = data[offset++];
            
            switch (edge_case % 5) {
                case 0:
                    // Test with empty dimension list but keepdim specified
                    torch::amin(input, torch::IntArrayRef{}, keepdim);
                    break;
                case 1:
                    // Test with all dimensions explicitly listed
                    if (tensor_rank > 0) {
                        std::vector<int64_t> all_dims;
                        for (int64_t i = 0; i < tensor_rank; ++i) {
                            all_dims.push_back(i);
                        }
                        torch::amin(input, all_dims, keepdim);
                    }
                    break;
                case 2:
                    // Test with negative dimension indexing
                    if (tensor_rank > 0) {
                        torch::amin(input, -1, keepdim);
                    }
                    break;
                case 3:
                    // Test chained operations
                    if (tensor_rank > 1) {
                        auto temp = torch::amin(input, 0, true);
                        torch::amin(temp, -1, false);
                    }
                    break;
                case 4:
                    // Test on non-contiguous tensor
                    if (tensor_rank > 0 && input.size(0) > 1) {
                        auto transposed = input.transpose(0, -1);
                        torch::amin(transposed, 0, keepdim);
                    }
                    break;
            }
        }
        
        // Test special tensor types if we have more data
        if (offset + 2 < size) {
            uint8_t special_case = data[offset++];
            
            switch (special_case % 4) {
                case 0:
                    // Test with tensor containing inf/nan for floating point types
                    if (input.is_floating_point()) {
                        auto special_input = input.clone();
                        if (special_input.numel() > 0) {
                            special_input.view(-1)[0] = std::numeric_limits<float>::infinity();
                            if (special_input.numel() > 1) {
                                special_input.view(-1)[1] = std::numeric_limits<float>::quiet_NaN();
                            }
                            torch::amin(special_input);
                        }
                    }
                    break;
                case 1:
                    // Test with zero-sized dimensions
                    if (tensor_rank > 0) {
                        auto shape = input.sizes().vec();
                        bool has_zero = false;
                        for (auto& s : shape) {
                            if (data[offset % size] & 1) {
                                s = 0;
                                has_zero = true;
                                break;
                            }
                        }
                        if (has_zero) {
                            auto zero_tensor = torch::empty(shape, input.options());
                            torch::amin(zero_tensor);
                        }
                    }
                    break;
                case 2:
                    // Test with very large tensor (if memory permits)
                    if ((data[offset % size] % 100) < 5) { // 5% chance to avoid OOM
                        try {
                            auto large_tensor = torch::ones({1000, 1000}, input.options());
                            torch::amin(large_tensor, 0);
                        } catch (const c10::Error& e) {
                            // Ignore OOM or other allocation errors
                        }
                    }
                    break;
                case 3:
                    // Test with complex dtypes if applicable
                    if (input.is_complex()) {
                        // amin should work with complex numbers (magnitude comparison)
                        torch::amin(input);
                    }
                    break;
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors (like invalid dimensions) are expected
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}