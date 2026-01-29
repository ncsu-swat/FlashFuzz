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
        
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters
        bool sorted = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        bool return_inverse = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        bool return_counts = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        
        // Get dimension parameter if there's enough data
        int64_t dim = 0;
        bool has_dim = false;
        if (offset < Size) {
            has_dim = (Data[offset++] % 2 == 0);
            if (has_dim && offset < Size) {
                dim = static_cast<int64_t>(Data[offset++]);
                
                // If tensor is not empty, ensure dim is within valid range
                if (input_tensor.dim() > 0) {
                    dim = dim % input_tensor.dim();
                } else {
                    has_dim = false; // Can't use dim on 0-d tensor
                }
            }
        }
        
        // Call torch::unique variants
        // torch::_unique2 is the main internal function that supports all options
        if (has_dim) {
            // unique_dim operates along a specific dimension
            try {
                auto result = torch::unique_dim(input_tensor, dim, sorted, return_inverse, return_counts);
                torch::Tensor output = std::get<0>(result);
                torch::Tensor inverse_indices = std::get<1>(result);
                torch::Tensor counts = std::get<2>(result);
                
                // Force evaluation
                (void)output.numel();
                if (return_inverse) {
                    (void)inverse_indices.numel();
                }
                if (return_counts) {
                    (void)counts.numel();
                }
            } catch (const c10::Error&) {
                // Shape/dimension errors are expected for some inputs
            }
        } else {
            // Flatten unique (no dimension specified)
            try {
                // Use _unique2 which returns (output, inverse, counts)
                auto result = torch::_unique2(input_tensor, sorted, return_inverse, return_counts);
                torch::Tensor output = std::get<0>(result);
                torch::Tensor inverse_indices = std::get<1>(result);
                torch::Tensor counts = std::get<2>(result);
                
                // Force evaluation
                (void)output.numel();
                if (return_inverse) {
                    (void)inverse_indices.numel();
                }
                if (return_counts) {
                    (void)counts.numel();
                }
            } catch (const c10::Error&) {
                // Expected errors for invalid inputs
            }
        }
        
        // Also test unique_consecutive variants for better coverage
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                auto result = torch::unique_consecutive(input_tensor, return_inverse, return_counts);
                torch::Tensor output = std::get<0>(result);
                (void)output.numel();
            } catch (const c10::Error&) {
                // Expected
            }
            
            if (has_dim && input_tensor.dim() > 0) {
                try {
                    auto result = torch::unique_dim_consecutive(input_tensor, dim, return_inverse, return_counts);
                    torch::Tensor output = std::get<0>(result);
                    (void)output.numel();
                } catch (const c10::Error&) {
                    // Expected
                }
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