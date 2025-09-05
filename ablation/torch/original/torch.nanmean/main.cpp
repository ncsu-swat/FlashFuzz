#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <limits>

// Helper to inject NaN values into floating point tensors
void injectNaNs(torch::Tensor& tensor, const uint8_t* data, size_t& offset, size_t size) {
    if (!tensor.is_floating_point() || tensor.numel() == 0) {
        return;
    }
    
    // Use fuzzer data to determine NaN injection pattern
    if (offset >= size) return;
    
    uint8_t nan_pattern = data[offset++];
    
    if (nan_pattern < 85) {  // ~33% chance: no NaNs
        return;
    } else if (nan_pattern < 170) {  // ~33% chance: some NaNs
        // Inject NaNs at random positions
        int64_t num_nans = (tensor.numel() * (nan_pattern % 50)) / 100;  // 0-50% NaNs
        for (int64_t i = 0; i < num_nans && offset < size; ++i) {
            if (offset >= size) break;
            int64_t idx = data[offset++] % tensor.numel();
            tensor.view(-1)[idx] = std::numeric_limits<float>::quiet_NaN();
        }
    } else {  // ~33% chance: all NaNs
        tensor.fill_(std::numeric_limits<float>::quiet_NaN());
    }
}

// Parse dimensions for reduction
std::vector<int64_t> parseDimensions(const uint8_t* data, size_t& offset, size_t size, int64_t tensor_dim) {
    if (offset >= size || tensor_dim == 0) {
        return {};  // No dimensions to reduce
    }
    
    uint8_t dim_selector = data[offset++];
    
    // Decide whether to use None (reduce all), single dim, or multiple dims
    if (dim_selector < 85) {  // ~33% chance: reduce all dimensions (None)
        return {};  // Empty vector means reduce all
    } else if (dim_selector < 170) {  // ~33% chance: single dimension
        if (offset >= size) return {0};
        int64_t dim = data[offset++] % tensor_dim;
        // Support negative indexing
        if (offset < size && data[offset] % 2 == 0) {
            dim = -tensor_dim + dim;
            offset++;
        }
        return {dim};
    } else {  // ~33% chance: multiple dimensions
        if (offset >= size) return {0};
        uint8_t num_dims = (data[offset++] % tensor_dim) + 1;
        std::vector<int64_t> dims;
        std::vector<bool> used(tensor_dim, false);
        
        for (uint8_t i = 0; i < num_dims && offset < size; ++i) {
            int64_t dim = data[offset++] % tensor_dim;
            if (!used[dim]) {
                used[dim] = true;
                // Support negative indexing
                if (offset < size && data[offset] % 2 == 0) {
                    dims.push_back(-tensor_dim + dim);
                    offset++;
                } else {
                    dims.push_back(dim);
                }
            }
        }
        return dims;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) {  // Need minimum bytes for basic parsing
        return 0;
    }
    
    try {
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(data, size, offset);
        
        // Inject NaN values for floating point tensors
        injectNaNs(input, data, offset, size);
        
        // Parse keepdim parameter
        bool keepdim = false;
        if (offset < size) {
            keepdim = (data[offset++] % 2) == 1;
        }
        
        // Parse dimensions to reduce
        std::vector<int64_t> dims = parseDimensions(data, offset, size, input.dim());
        
        // Parse dtype option (for output casting)
        c10::optional<torch::ScalarType> dtype_opt = c10::nullopt;
        if (offset < size) {
            uint8_t use_dtype = data[offset++];
            if (use_dtype > 200) {  // ~20% chance to specify dtype
                if (offset < size) {
                    // Only use floating point types for nanmean
                    std::vector<torch::ScalarType> float_types = {
                        torch::kFloat, torch::kDouble, torch::kHalf, torch::kBFloat16
                    };
                    uint8_t dtype_idx = data[offset++] % float_types.size();
                    dtype_opt = float_types[dtype_idx];
                }
            }
        }
        
        // Test different invocation patterns
        if (offset < size) {
            uint8_t test_variant = data[offset++];
            
            if (test_variant < 64) {  // Test with no dimensions (reduce all)
                torch::Tensor result;
                if (dtype_opt.has_value()) {
                    result = torch::nanmean(input, /*dim=*/{}, /*keepdim=*/keepdim, 
                                          /*dtype=*/dtype_opt.value());
                } else {
                    result = torch::nanmean(input, /*dim=*/{}, /*keepdim=*/keepdim);
                }
                
                // Verify result properties
                if (keepdim && input.dim() > 0) {
                    if (result.dim() != input.dim()) {
                        std::cerr << "Unexpected dimension change with keepdim=true" << std::endl;
                    }
                }
                
            } else if (test_variant < 128 && !dims.empty()) {  // Test with single dimension
                torch::Tensor result;
                if (dims.size() == 1) {
                    if (dtype_opt.has_value()) {
                        result = torch::nanmean(input, /*dim=*/dims[0], /*keepdim=*/keepdim,
                                              /*dtype=*/dtype_opt.value());
                    } else {
                        result = torch::nanmean(input, /*dim=*/dims[0], /*keepdim=*/keepdim);
                    }
                }
                
            } else if (test_variant < 192 && !dims.empty()) {  // Test with multiple dimensions
                torch::Tensor result;
                if (dtype_opt.has_value()) {
                    result = torch::nanmean(input, /*dim=*/dims, /*keepdim=*/keepdim,
                                          /*dtype=*/dtype_opt.value());
                } else {
                    result = torch::nanmean(input, /*dim=*/dims, /*keepdim=*/keepdim);
                }
                
            } else {  // Test edge cases
                // Test with out parameter
                torch::Tensor out_tensor;
                
                // Calculate expected output shape
                std::vector<int64_t> out_shape = input.sizes().vec();
                if (!dims.empty()) {
                    if (keepdim) {
                        for (auto dim : dims) {
                            int64_t actual_dim = dim < 0 ? input.dim() + dim : dim;
                            if (actual_dim >= 0 && actual_dim < input.dim()) {
                                out_shape[actual_dim] = 1;
                            }
                        }
                    } else {
                        // Remove dimensions (more complex, skip for simplicity)
                        // Just create a tensor with some shape
                        out_tensor = torch::empty({1}, input.options());
                    }
                } else if (keepdim) {
                    std::fill(out_shape.begin(), out_shape.end(), 1);
                    out_tensor = torch::empty(out_shape, input.options());
                } else {
                    out_tensor = torch::empty({}, input.options());  // scalar
                }
                
                if (out_tensor.defined()) {
                    torch::nanmean_out(out_tensor, input, dims, keepdim);
                }
            }
        }
        
        // Additional edge case testing
        if (offset < size && data[offset++] > 200) {
            // Test with empty tensor
            torch::Tensor empty_tensor = torch::empty({0}, input.options());
            try {
                auto result = torch::nanmean(empty_tensor);
            } catch (...) {
                // Expected to potentially fail on empty tensor
            }
            
            // Test with all-NaN tensor
            if (input.is_floating_point()) {
                torch::Tensor all_nan = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
                auto result = torch::nanmean(all_nan);
                // Result should be NaN when all values are NaN
            }
            
            // Test with scalar tensor
            torch::Tensor scalar = torch::tensor(3.14f);
            auto scalar_result = torch::nanmean(scalar);
            
            // Test with high-dimensional tensor
            if (offset + 5 < size) {
                std::vector<int64_t> high_dims;
                for (int i = 0; i < 5 && offset < size; ++i) {
                    high_dims.push_back((data[offset++] % 3) + 1);
                }
                try {
                    torch::Tensor high_dim = torch::randn(high_dims);
                    auto result = torch::nanmean(high_dim);
                } catch (...) {
                    // May fail due to memory constraints
                }
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions
        return -1;
    }
    
    return 0;
}