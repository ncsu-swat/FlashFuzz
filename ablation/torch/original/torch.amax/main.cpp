#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 4) {
        // Need minimum bytes for: tensor creation + dim selection + keepdim flag
        return 0;
    }

    try
    {
        size_t offset = 0;
        
        // Create input tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(data, size, offset);
        
        // If no more data available for parameters, use defaults
        if (offset >= size) {
            // Test with default parameters (reduce last dim, keepdim=false)
            if (input.dim() > 0) {
                torch::Tensor result = torch::amax(input, -1, false);
            }
            return 0;
        }
        
        // Parse reduction dimension(s)
        uint8_t dim_mode = (offset < size) ? data[offset++] : 0;
        
        // Parse keepdim flag
        bool keepdim = (offset < size) ? (data[offset++] & 1) : false;
        
        // Determine whether to use single dim or multiple dims
        bool use_multiple_dims = (dim_mode & 0x80) != 0;
        
        if (input.dim() == 0) {
            // Scalar tensor - amax should work but dimension reduction is not applicable
            torch::Tensor result = torch::amax(input, {}, keepdim);
        }
        else if (use_multiple_dims && input.dim() > 1) {
            // Test with multiple dimensions
            uint8_t num_dims_to_reduce = ((dim_mode & 0x7F) % input.dim()) + 1;
            num_dims_to_reduce = std::min(static_cast<int>(num_dims_to_reduce), input.dim());
            
            std::vector<int64_t> dims;
            std::vector<bool> used(input.dim(), false);
            
            for (int i = 0; i < num_dims_to_reduce; ++i) {
                uint8_t dim_selector = (offset < size) ? data[offset++] : i;
                int64_t dim = dim_selector % input.dim();
                
                // Avoid duplicate dimensions
                int attempts = 0;
                while (used[dim] && attempts < input.dim()) {
                    dim = (dim + 1) % input.dim();
                    attempts++;
                }
                
                if (!used[dim]) {
                    dims.push_back(dim);
                    used[dim] = true;
                }
            }
            
            if (!dims.empty()) {
                torch::Tensor result = torch::amax(input, dims, keepdim);
                
                // Verify output shape
                if (keepdim) {
                    // Check that reduced dimensions have size 1
                    for (auto d : dims) {
                        if (result.size(d) != 1) {
                            std::cerr << "Unexpected: keepdim=true but dim " << d 
                                     << " has size " << result.size(d) << std::endl;
                        }
                    }
                }
            }
        }
        else {
            // Test with single dimension
            int64_t dim;
            
            if (input.dim() == 1) {
                // Only one dimension available
                dim = 0;
            } else {
                // Select dimension based on fuzzer input
                uint8_t dim_selector = dim_mode & 0x7F;
                
                // Test both positive and negative indexing
                if (dim_selector & 0x40) {
                    // Use negative indexing
                    dim = -(static_cast<int64_t>((dim_selector & 0x3F) % input.dim()) + 1);
                } else {
                    // Use positive indexing
                    dim = dim_selector % input.dim();
                }
            }
            
            torch::Tensor result = torch::amax(input, dim, keepdim);
            
            // Additional operations to increase coverage
            if (offset < size && (data[offset++] & 1)) {
                // Test with output tensor
                torch::Tensor out = torch::empty_like(result);
                torch::amax_out(out, input, dim, keepdim);
                
                // Verify in-place operation worked correctly
                if (!torch::allclose(result, out, 1e-5, 1e-8)) {
                    std::cerr << "amax_out produced different result than amax" << std::endl;
                }
            }
        }
        
        // Test edge cases based on remaining fuzzer data
        if (offset < size) {
            uint8_t edge_case = data[offset++];
            
            switch (edge_case & 0x7) {
                case 0:
                    // Test reducing all dimensions
                    if (input.dim() > 0) {
                        std::vector<int64_t> all_dims;
                        for (int i = 0; i < input.dim(); ++i) {
                            all_dims.push_back(i);
                        }
                        torch::Tensor result = torch::amax(input, all_dims, keepdim);
                    }
                    break;
                    
                case 1:
                    // Test with empty dimension list (should return input unchanged)
                    {
                        torch::Tensor result = torch::amax(input, {}, keepdim);
                        if (!torch::equal(result, input)) {
                            std::cerr << "Empty dim list should return input unchanged" << std::endl;
                        }
                    }
                    break;
                    
                case 2:
                    // Test with special values if float type
                    if (input.is_floating_point()) {
                        // Add some special values
                        if (input.numel() > 0) {
                            input.view(-1)[0] = std::numeric_limits<float>::infinity();
                            if (input.numel() > 1) {
                                input.view(-1)[1] = -std::numeric_limits<float>::infinity();
                            }
                            if (input.numel() > 2) {
                                input.view(-1)[2] = std::numeric_limits<float>::quiet_NaN();
                            }
                        }
                        
                        if (input.dim() > 0) {
                            torch::Tensor result = torch::amax(input, 0, keepdim);
                        }
                    }
                    break;
                    
                case 3:
                    // Test with non-contiguous tensor
                    if (input.dim() >= 2 && input.size(0) > 1 && input.size(1) > 1) {
                        torch::Tensor transposed = input.transpose(0, 1);
                        if (!transposed.is_contiguous()) {
                            torch::Tensor result = torch::amax(transposed, 0, keepdim);
                        }
                    }
                    break;
                    
                case 4:
                    // Test comparison with max (they should give same values, different gradients)
                    if (input.dim() > 0 && input.requires_grad()) {
                        torch::Tensor amax_result = torch::amax(input, 0, keepdim);
                        auto max_result = torch::max(input, 0, keepdim);
                        torch::Tensor max_values = std::get<0>(max_result);
                        
                        if (!torch::allclose(amax_result, max_values, 1e-5, 1e-8)) {
                            std::cerr << "amax and max values differ unexpectedly" << std::endl;
                        }
                    }
                    break;
                    
                case 5:
                    // Test with sliced tensor
                    if (input.dim() > 0 && input.size(0) > 2) {
                        torch::Tensor sliced = input.slice(0, 1, -1);
                        torch::Tensor result = torch::amax(sliced, 0, keepdim);
                    }
                    break;
                    
                case 6:
                    // Test with reshaped tensor
                    if (input.numel() > 0) {
                        std::vector<int64_t> new_shape;
                        int64_t remaining = input.numel();
                        
                        // Create a different valid shape
                        while (remaining > 1) {
                            int64_t factor = 2;
                            if (remaining % factor == 0) {
                                new_shape.push_back(factor);
                                remaining /= factor;
                            } else {
                                new_shape.push_back(remaining);
                                break;
                            }
                        }
                        
                        if (!new_shape.empty()) {
                            torch::Tensor reshaped = input.reshape(new_shape);
                            torch::Tensor result = torch::amax(reshaped, 0, keepdim);
                        }
                    }
                    break;
                    
                case 7:
                    // Test with complex tensors (if applicable)
                    if (input.dtype() == torch::kComplexFloat || input.dtype() == torch::kComplexDouble) {
                        // amax should work with complex tensors (comparing by absolute value)
                        if (input.dim() > 0) {
                            torch::Tensor result = torch::amax(input, 0, keepdim);
                        }
                    }
                    break;
            }
        }
        
        // Test chaining operations
        if (offset < size && (data[offset++] & 1)) {
            if (input.dim() >= 2) {
                // Chain multiple amax operations
                torch::Tensor result1 = torch::amax(input, 0, true);
                torch::Tensor result2 = torch::amax(result1, 1, false);
            }
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected for invalid operations
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}