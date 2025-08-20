#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for unsafe_split
        // We need at least 2 bytes for sections and dim
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Get number of sections
        int64_t sections = static_cast<int64_t>(Data[offset++]);
        if (sections == 0) {
            sections = 1; // Ensure at least 1 section
        }
        
        // Get dimension to split along
        int64_t dim = 0;
        if (tensor.dim() > 0) {
            // Use the byte to select a dimension
            dim = static_cast<int64_t>(Data[offset++]) % tensor.dim();
        }
        
        // Apply torch.unsafe_split
        std::vector<torch::Tensor> result;
        try {
            result = torch::unsafe_split(tensor, sections, dim);
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors but don't discard the input
            return 0;
        }
        
        // Verify the result (optional)
        if (!result.empty()) {
            // Check if we can concatenate the split tensors back
            try {
                torch::Tensor reconstructed = torch::cat(result, dim);
                
                // Verify sizes match
                if (reconstructed.sizes() != tensor.sizes()) {
                    // This is a potential issue, but we'll continue
                }
            } catch (const c10::Error& e) {
                // Concatenation failed, but we'll continue
            }
        }
        
        // Try another variant with split_with_sizes if we have more data
        if (offset + 1 < Size && tensor.dim() > 0) {
            // Create a vector of split sizes
            std::vector<int64_t> split_sizes;
            int64_t dim_size = tensor.size(dim);
            
            // Use remaining bytes to determine split sizes
            int64_t remaining_size = dim_size;
            size_t max_splits_value = static_cast<size_t>(Data[offset++]);
            size_t max_splits = std::min(max_splits_value, Size - offset);
            
            for (size_t i = 0; i < max_splits && offset < Size && remaining_size > 0; ++i) {
                // Get a size between 0 and remaining_size
                int64_t split_size = static_cast<int64_t>(Data[offset++] % (remaining_size + 1));
                if (split_size == 0) split_size = 1; // Avoid zero-sized splits
                
                split_sizes.push_back(split_size);
                remaining_size -= split_size;
            }
            
            // Add the remaining size as the last split if needed
            if (remaining_size > 0) {
                split_sizes.push_back(remaining_size);
            }
            
            // Apply torch.unsafe_split_with_sizes
            try {
                std::vector<torch::Tensor> result_sizes = torch::unsafe_split_with_sizes(tensor, split_sizes, dim);
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors but don't discard the input
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