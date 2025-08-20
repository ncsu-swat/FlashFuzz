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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, L) for AdaptiveMaxPool1d
        // If not, reshape it to have at least 3 dimensions
        if (input.dim() < 1) {
            // For 0-dim tensor, reshape to [1, 1, 1]
            input = input.reshape({1, 1, 1});
        } else if (input.dim() == 1) {
            // For 1-dim tensor, treat as single feature with batch size 1
            input = input.reshape({1, 1, input.size(0)});
        } else if (input.dim() == 2) {
            // For 2-dim tensor, treat as batch size x features
            input = input.reshape({input.size(0), input.size(1), 1});
        }
        
        // Get output size from remaining data
        int64_t output_size = 1; // Default
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&output_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure output_size is reasonable (can be negative to test error cases)
            if (output_size > 100) {
                output_size = output_size % 100 + 1;
            }
        }
        
        // Create AdaptiveMaxPool1d module
        torch::nn::AdaptiveMaxPool1d pool(output_size);
        
        // Apply the operation
        auto output = pool(input);
        
        // Try to access the indices if available
        if (offset < Size) {
            bool get_indices = Data[offset++] % 2 == 0;
            if (get_indices) {
                // Use functional interface to get indices
                try {
                    auto result = torch::nn::functional::adaptive_max_pool1d(input, 
                        torch::nn::functional::AdaptiveMaxPool1dFuncOptions(output_size));
                    // Use the result to ensure it's computed
                    auto dummy = result.sum();
                } catch (...) {
                    // Ignore errors
                }
            }
        }
        
        // Try different data types
        if (offset < Size && input.dim() >= 3) {
            // Convert to different dtype and try again
            auto dtype_selector = Data[offset++] % 4;
            torch::ScalarType new_dtype;
            
            switch (dtype_selector) {
                case 0: new_dtype = torch::kFloat; break;
                case 1: new_dtype = torch::kDouble; break;
                case 2: new_dtype = torch::kHalf; break;
                case 3: new_dtype = torch::kBFloat16; break;
                default: new_dtype = torch::kFloat;
            }
            
            // Only convert if the dtype is different
            if (input.scalar_type() != new_dtype) {
                try {
                    auto converted_input = input.to(new_dtype);
                    auto converted_output = pool(converted_input);
                } catch (...) {
                    // Ignore conversion errors
                }
            }
        }
        
        // Try with different output sizes
        if (offset + sizeof(int64_t) <= Size) {
            int64_t alt_output_size;
            std::memcpy(&alt_output_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Allow negative values to test error handling
            if (alt_output_size > 100) {
                alt_output_size = alt_output_size % 100 + 1;
            }
            
            torch::nn::AdaptiveMaxPool1d alt_pool(alt_output_size);
            try {
                auto alt_output = alt_pool(input);
            } catch (...) {
                // Ignore errors from invalid output sizes
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}