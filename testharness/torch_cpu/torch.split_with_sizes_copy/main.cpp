#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <algorithm>      // For std::max

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
                dim = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
            }
            
            // Get the size of the dimension we're splitting
            int64_t dim_size = input_tensor.size(dim);
            
            if (dim_size == 0) {
                // Can't split a dimension of size 0 meaningfully
                return 0;
            }
            
            // Create split_sizes vector
            std::vector<int64_t> split_sizes;
            
            // Determine number of splits (1-8 splits)
            uint8_t num_splits = 1;
            if (offset < Size) {
                num_splits = (Data[offset++] % 8) + 1;
            }
            
            // Strategy selection: sometimes generate valid splits, sometimes invalid
            uint8_t strategy = 0;
            if (offset < Size) {
                strategy = Data[offset++] % 3;
            }
            
            if (strategy == 0) {
                // Generate split sizes that sum exactly to dim_size (valid case)
                int64_t remaining = dim_size;
                for (uint8_t i = 0; i < num_splits - 1 && offset < Size && remaining > 1; ++i) {
                    int64_t max_split = remaining - (num_splits - i - 1); // Leave at least 1 for remaining splits
                    if (max_split <= 0) max_split = 1;
                    int64_t split_size = (static_cast<int64_t>(Data[offset++]) % max_split) + 1;
                    split_sizes.push_back(split_size);
                    remaining -= split_size;
                }
                // Add the remaining as the last split
                if (remaining > 0) {
                    split_sizes.push_back(remaining);
                }
            } else if (strategy == 1) {
                // Generate arbitrary split sizes (may be invalid - tests error handling)
                for (uint8_t i = 0; i < num_splits && offset < Size; ++i) {
                    int64_t split_size = static_cast<int64_t>(Data[offset++]);
                    split_sizes.push_back(split_size);
                }
            } else {
                // Generate split sizes that intentionally don't sum correctly
                for (uint8_t i = 0; i < num_splits && offset < Size; ++i) {
                    int64_t split_size = (static_cast<int64_t>(Data[offset++]) % (dim_size + 1)) + 1;
                    split_sizes.push_back(split_size);
                }
            }
            
            // If we have no split sizes, add at least one
            if (split_sizes.empty()) {
                split_sizes.push_back(dim_size);
            }
            
            // Call split_with_sizes_copy
            try {
                std::vector<torch::Tensor> result = torch::split_with_sizes_copy(input_tensor, split_sizes, dim);
                
                // Verify result tensors exist and have correct properties
                for (const auto& t : result) {
                    (void)t.numel();
                    (void)t.dim();
                }
            } catch (const c10::Error& e) {
                // PyTorch-specific exceptions are expected for invalid inputs
            }
        } else {
            // For 0-dim tensors, try splitting - this should throw
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