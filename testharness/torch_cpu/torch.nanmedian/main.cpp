#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a dimension value from the remaining data if available
        int64_t dim = 0;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Extract keepdim boolean if data available
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x1);
            offset++;
        }
        
        // Variant 1: Basic nanmedian (no arguments) - returns scalar tensor
        try {
            torch::Tensor result1 = torch::nanmedian(input);
            (void)result1;
        } catch (...) {
            // Silently catch expected failures (e.g., empty tensor)
        }
        
        // Variant 2: nanmedian with dimension
        if (input.dim() > 0) {
            try {
                // Ensure dim is within valid range for the tensor
                int64_t valid_dim = dim % input.dim();
                
                // Handle negative dim values properly
                if (valid_dim < 0) {
                    valid_dim += input.dim();
                }
                
                // Call nanmedian with dimension - returns tuple of (values, indices)
                auto result2 = torch::nanmedian(input, valid_dim, keepdim);
                
                // Access the values and indices to ensure they're computed
                torch::Tensor values = std::get<0>(result2);
                torch::Tensor indices = std::get<1>(result2);
                (void)values;
                (void)indices;
            } catch (...) {
                // Silently catch expected failures (shape issues, etc.)
            }
        }
        
        // Variant 3: Test with out parameter
        if (input.dim() > 0 && input.numel() > 0) {
            try {
                int64_t valid_dim = dim % input.dim();
                if (valid_dim < 0) {
                    valid_dim += input.dim();
                }
                
                // Create output tensors with correct shapes
                std::vector<int64_t> out_shape;
                auto in_sizes = input.sizes().vec();
                
                if (keepdim) {
                    out_shape = in_sizes;
                    out_shape[valid_dim] = 1;
                } else {
                    for (int64_t i = 0; i < static_cast<int64_t>(in_sizes.size()); i++) {
                        if (i != valid_dim) {
                            out_shape.push_back(in_sizes[i]);
                        }
                    }
                    // Handle scalar case when reducing 1D tensor
                    if (out_shape.empty()) {
                        out_shape.push_back(1);
                    }
                }
                
                // Create output tensors - values should match input dtype, indices are Long
                torch::Tensor values_out = torch::empty(out_shape, input.options());
                torch::Tensor indices_out = torch::empty(out_shape, input.options().dtype(torch::kLong));
                
                // Call nanmedian_out
                torch::nanmedian_out(values_out, indices_out, input, valid_dim, keepdim);
                (void)values_out;
                (void)indices_out;
            } catch (...) {
                // Silently catch expected failures
            }
        }
        
        // Variant 4: Test with different tensor types
        if (input.dim() > 0) {
            try {
                // Try with float tensor explicitly
                torch::Tensor float_input = input.to(torch::kFloat);
                torch::Tensor result = torch::nanmedian(float_input);
                (void)result;
            } catch (...) {
                // Silently catch
            }
            
            try {
                // Try with double tensor
                torch::Tensor double_input = input.to(torch::kDouble);
                torch::Tensor result = torch::nanmedian(double_input);
                (void)result;
            } catch (...) {
                // Silently catch
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}