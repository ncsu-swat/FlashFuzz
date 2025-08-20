#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation and split sizes
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have some data left for split sizes
        if (offset >= Size) {
            return 0;
        }
        
        // Determine dimension to split along
        int64_t dim = 0;
        if (input_tensor.dim() > 0) {
            // Read a byte for the dimension
            if (offset < Size) {
                dim = static_cast<int64_t>(Data[offset++]) % std::max(static_cast<int64_t>(1), input_tensor.dim());
            }
            
            // Create split_sizes vector
            std::vector<int64_t> split_sizes;
            
            // Determine number of splits (1-8 splits)
            uint8_t num_splits = 1;
            if (offset < Size) {
                num_splits = (Data[offset++] % 8) + 1;
            }
            
            // Get the size of the dimension we're splitting
            int64_t dim_size = input_tensor.dim() > 0 ? input_tensor.size(dim) : 0;
            
            // Generate split sizes
            for (uint8_t i = 0; i < num_splits && offset < Size; ++i) {
                // Use the next byte as a split size
                int64_t split_size = static_cast<int64_t>(Data[offset++]);
                
                // Allow negative split sizes to test error handling
                split_sizes.push_back(split_size);
            }
            
            // If we have no split sizes, add at least one
            if (split_sizes.empty()) {
                split_sizes.push_back(1);
            }
            
            // Call split_with_sizes_copy
            try {
                std::vector<torch::Tensor> result = torch::split_with_sizes_copy(input_tensor, split_sizes, dim);
            } catch (const c10::Error& e) {
                // PyTorch-specific exceptions are expected for invalid inputs
            }
        } else {
            // For 0-dim tensors, try splitting with a single size
            std::vector<int64_t> split_sizes = {1};
            try {
                std::vector<torch::Tensor> result = torch::split_with_sizes_copy(input_tensor, split_sizes, 0);
            } catch (const c10::Error& e) {
                // PyTorch-specific exceptions are expected for invalid inputs
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