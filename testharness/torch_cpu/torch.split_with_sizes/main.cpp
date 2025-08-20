#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse number of splits
        if (offset + sizeof(int32_t) > Size) {
            return 0;
        }
        
        int32_t num_splits_raw;
        std::memcpy(&num_splits_raw, Data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        
        // Ensure we have at least 1 split and not too many
        int num_splits = std::abs(num_splits_raw) % 10 + 1;
        
        // Parse split sizes
        std::vector<int64_t> split_sizes;
        for (int i = 0; i < num_splits && offset + sizeof(int32_t) <= Size; ++i) {
            int32_t split_size_raw;
            std::memcpy(&split_size_raw, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            
            // Allow negative split sizes to test error handling
            split_sizes.push_back(split_size_raw);
        }
        
        // If we couldn't read enough split sizes, generate some
        if (split_sizes.empty()) {
            split_sizes.push_back(1);
        }
        
        // Parse dimension to split along
        int64_t dim = 0;
        if (offset + sizeof(int32_t) <= Size) {
            int32_t dim_raw;
            std::memcpy(&dim_raw, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            
            // Allow negative dimensions to test error handling
            dim = dim_raw;
        }
        
        // Apply the split_with_sizes operation
        std::vector<torch::Tensor> result;
        try {
            result = torch::split_with_sizes(input_tensor, split_sizes, dim);
            
            // Perform some operations on the result to ensure it's valid
            for (const auto& tensor : result) {
                auto sum = tensor.sum();
                auto mean = tensor.mean();
                
                // Force evaluation of the tensors
                sum.item<float>();
                if (tensor.numel() > 0) {
                    mean.item<float>();
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}