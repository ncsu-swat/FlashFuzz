#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>
#include <vector>

// --- Fuzzer Entry Point ---
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
        
        // Create input tensor - use float for numerical stability
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float for logsumexp operation
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Handle scalar tensors
        if (input.dim() == 0) {
            // For scalar, logsumexp with dim=0 should work or we skip
            try {
                torch::Tensor result = torch::logsumexp(input.unsqueeze(0), 0);
            } catch (...) {
                // Expected for some edge cases
            }
            return 0;
        }
        
        // Parse dim parameter if there's data left
        int64_t dim = 0;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure dim is within valid range [-ndim, ndim-1]
            int64_t ndim = input.dim();
            dim = ((dim % ndim) + ndim) % ndim;
        }
        
        // Parse keepdim parameter if there's data left
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x01);
            offset++;
        }
        
        // Variant 1: Basic logsumexp with single dimension
        torch::Tensor result1 = torch::logsumexp(input, dim, keepdim);
        
        // Variant 2: If we have a multi-dimensional tensor, try with a list of dimensions
        if (input.dim() > 1) {
            try {
                int64_t dim2 = 0;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&dim2, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                }
                
                int64_t ndim = input.dim();
                dim2 = ((dim2 % ndim) + ndim) % ndim;
                
                // Ensure dim2 is different from dim
                if (dim2 == dim) {
                    dim2 = (dim2 + 1) % ndim;
                }
                
                // Sort dimensions for consistent behavior
                std::vector<int64_t> dims;
                if (dim < dim2) {
                    dims = {dim, dim2};
                } else {
                    dims = {dim2, dim};
                }
                torch::Tensor result2 = torch::logsumexp(input, dims, keepdim);
            } catch (...) {
                // Silently catch dimension-related errors
            }
        }
        
        // Variant 3: Reduce over all dimensions
        if (input.dim() > 0) {
            try {
                std::vector<int64_t> all_dims;
                for (int64_t i = 0; i < input.dim(); i++) {
                    all_dims.push_back(i);
                }
                torch::Tensor result3 = torch::logsumexp(input, all_dims, keepdim);
            } catch (...) {
                // Silently catch errors
            }
        }
        
        // Variant 4: Test with different dtypes
        if (offset < Size) {
            try {
                uint8_t dtype_selector = Data[offset] % 3;
                offset++;
                
                torch::Tensor typed_input;
                switch (dtype_selector) {
                    case 0:
                        typed_input = input.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_input = input.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_input = input.to(torch::kFloat16);
                        break;
                    default:
                        typed_input = input;
                }
                torch::Tensor result4 = torch::logsumexp(typed_input, dim, keepdim);
            } catch (...) {
                // Silently catch dtype conversion errors
            }
        }
        
        // Variant 5: Test with contiguous vs non-contiguous tensor
        if (input.dim() >= 2) {
            try {
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor result5 = torch::logsumexp(transposed, 0, keepdim);
            } catch (...) {
                // Silently catch errors
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