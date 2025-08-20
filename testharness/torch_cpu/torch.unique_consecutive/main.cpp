#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        if (offset + 1 < Size) {
            return_inverse = Data[offset++] & 0x1;
        }
        
        if (offset + 1 < Size) {
            return_counts = Data[offset++] & 0x1;
        }
        
        if (offset + 1 < Size) {
            // Determine if we should use a dimension parameter
            bool use_dim = Data[offset++] & 0x1;
            
            if (use_dim && offset + sizeof(int64_t) <= Size) {
                int64_t dim_value;
                std::memcpy(&dim_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // If tensor is not empty, ensure dim is within valid range
                if (input_tensor.dim() > 0) {
                    // Allow negative dimensions to test handling of negative indices
                    dim = dim_value;
                }
            }
        }
        
        // Call unique_consecutive with different parameter combinations
        if (!return_inverse && !return_counts) {
            // Basic case: just return unique elements
            torch::Tensor output = torch::unique_consecutive(input_tensor, false, false, dim);
        } 
        else if (return_inverse && !return_counts) {
            // Return unique elements and inverse indices
            auto result = torch::unique_consecutive(input_tensor, return_inverse, false, dim);
            auto output = std::get<0>(result);
            auto inverse_indices = std::get<1>(result);
        } 
        else if (!return_inverse && return_counts) {
            // Return unique elements and counts
            auto result = torch::unique_consecutive(input_tensor, false, return_counts, dim);
            auto output = std::get<0>(result);
            auto counts = std::get<2>(result);
        } 
        else {
            // Return all: unique elements, inverse indices, and counts
            auto result = torch::unique_consecutive(input_tensor, return_inverse, return_counts, dim);
            auto output = std::get<0>(result);
            auto inverse_indices = std::get<1>(result);
            auto counts = std::get<2>(result);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}