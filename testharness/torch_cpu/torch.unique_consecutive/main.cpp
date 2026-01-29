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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for unique_consecutive
        bool return_inverse = false;
        bool return_counts = false;
        c10::optional<int64_t> dim = c10::nullopt;
        
        // Use remaining bytes to determine parameters if available
        if (offset + 1 <= Size) {
            return_inverse = Data[offset++] & 0x1;
        }
        
        if (offset + 1 <= Size) {
            return_counts = Data[offset++] & 0x1;
        }
        
        if (offset + 1 <= Size) {
            // Determine if we should use a dimension parameter
            bool use_dim = Data[offset++] & 0x1;
            
            if (use_dim && offset + sizeof(int64_t) <= Size) {
                int64_t dim_value;
                std::memcpy(&dim_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // If tensor has dimensions, use the dim parameter
                if (input_tensor.dim() > 0) {
                    // Clamp to valid range to improve meaningful coverage
                    // Allow some out-of-range values to test error handling
                    if (dim_value > 100) {
                        dim_value = dim_value % (input_tensor.dim() + 2) - 1;
                    } else if (dim_value < -100) {
                        dim_value = -((-dim_value) % (input_tensor.dim() + 2)) - 1;
                    }
                    dim = dim_value;
                }
            }
        }
        
        // Inner try-catch for expected failures (invalid dimensions, etc.)
        try {
            // Call unique_consecutive with different parameter combinations
            // The function always returns a tuple of (output, inverse_indices, counts)
            // even if return_inverse/return_counts are false (those tensors will be empty)
            auto result = torch::unique_consecutive(input_tensor, return_inverse, return_counts, dim);
            
            // Access the results to ensure they're computed
            torch::Tensor output = std::get<0>(result);
            
            if (return_inverse) {
                torch::Tensor inverse_indices = std::get<1>(result);
                // Verify inverse indices are valid if we have output
                if (output.numel() > 0 && inverse_indices.numel() > 0) {
                    (void)inverse_indices.sum();
                }
            }
            
            if (return_counts) {
                torch::Tensor counts = std::get<2>(result);
                // Verify counts are valid
                if (counts.numel() > 0) {
                    (void)counts.sum();
                }
            }
            
            // Force computation
            if (output.numel() > 0) {
                (void)output.sum();
            }
        }
        catch (const c10::Error& e) {
            // Expected errors (invalid dim, etc.) - catch silently
        }
        catch (const std::runtime_error& e) {
            // Expected runtime errors - catch silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}