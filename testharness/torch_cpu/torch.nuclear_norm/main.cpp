#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Nuclear norm requires at least a 2D tensor (matrix)
        // If tensor is less than 2D, expand it
        if (input.dim() < 2) {
            if (input.dim() == 0) {
                input = input.unsqueeze(0).unsqueeze(0);
            } else if (input.dim() == 1) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract parameters for nuclear_norm
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Determine which variant to call
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 3;
        }
        
        torch::Tensor result;
        
        try {
            if (variant == 0) {
                // Call basic nuclear_norm (treats input as 2D matrix or batch of matrices)
                result = torch::nuclear_norm(input, keepdim);
            } else {
                // Call with explicit dim parameter (must be exactly 2 dimensions)
                int64_t ndim = input.dim();
                if (ndim >= 2) {
                    int64_t dim1 = 0, dim2 = 1;
                    
                    if (offset + 1 < Size) {
                        // Select two different dimensions
                        dim1 = Data[offset++] % ndim;
                        dim2 = Data[offset++] % ndim;
                        
                        // Ensure dimensions are different
                        if (dim1 == dim2) {
                            dim2 = (dim1 + 1) % ndim;
                        }
                    } else {
                        // Default to last two dimensions
                        dim1 = ndim - 2;
                        dim2 = ndim - 1;
                    }
                    
                    result = torch::nuclear_norm(input, torch::IntArrayRef({dim1, dim2}), keepdim);
                } else {
                    // Fallback to basic version
                    result = torch::nuclear_norm(input, keepdim);
                }
            }
            
            // Access result to ensure computation is performed
            if (result.defined()) {
                // Use sum() to handle both scalar and non-scalar results
                volatile auto sum_val = result.sum().item<float>();
                (void)sum_val;
            }
        }
        catch (const c10::Error &e) {
            // Expected errors for invalid tensor configurations (shape mismatches, etc.)
            // Silently ignore these
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}