#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte to determine number of tensors
        if (Size < 1) {
            return 0;
        }
        
        // Determine number of tensors to concatenate (1-8)
        uint8_t num_tensors = (Data[offset++] % 8) + 1;
        
        // Create a vector to hold our tensors
        std::vector<torch::Tensor> tensors;
        
        // Create tensors
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
            } catch (const std::exception& e) {
                // If we can't create a tensor, just continue with what we have
                break;
            }
        }
        
        // Need at least one tensor to proceed
        if (tensors.empty()) {
            return 0;
        }
        
        // Determine dimension to concatenate along
        int64_t dim = 0;
        if (offset < Size) {
            // Get the rank of the first tensor
            int64_t max_dim = tensors[0].dim() - 1;
            if (max_dim >= 0) {
                // Allow negative dimensions for testing edge cases
                dim = static_cast<int64_t>(Data[offset++]);
                // Don't clamp the dimension to allow testing out-of-bounds cases
            }
        }
        
        // Try to concatenate the tensors
        torch::Tensor result = torch::cat(tensors, dim);
        
        // Try some additional operations on the result to increase coverage
        if (!result.sizes().empty()) {
            // Try to sum the result
            torch::Tensor sum = result.sum();
            
            // Try to reshape if possible
            if (result.numel() > 0) {
                try {
                    torch::Tensor reshaped = result.reshape({-1});
                } catch (...) {
                    // Ignore reshape errors
                }
            }
            
            // Try slicing if possible
            if (result.dim() > 0 && result.size(0) > 0) {
                try {
                    torch::Tensor sliced = result.slice(0, 0, result.size(0) / 2);
                } catch (...) {
                    // Ignore slicing errors
                }
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