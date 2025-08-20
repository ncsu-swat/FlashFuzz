#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for as_strided_
        // We need: size, stride, storage_offset
        
        // Parse number of dimensions for the new size
        uint8_t size_rank = 0;
        if (offset < Size) {
            size_rank = fuzzer_utils::parseRank(Data[offset++]);
        }
        
        // Parse the new size
        std::vector<int64_t> new_size;
        if (offset < Size && size_rank > 0) {
            new_size = fuzzer_utils::parseShape(Data, offset, Size, size_rank);
        }
        
        // Parse number of dimensions for the new stride
        uint8_t stride_rank = 0;
        if (offset < Size) {
            stride_rank = fuzzer_utils::parseRank(Data[offset++]);
        }
        
        // Parse the new stride
        std::vector<int64_t> new_stride;
        if (offset < Size && stride_rank > 0) {
            new_stride = fuzzer_utils::parseShape(Data, offset, Size, stride_rank);
        }
        
        // Parse storage offset
        int64_t storage_offset = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&storage_offset, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Make sure size and stride have the same number of dimensions
        if (new_size.size() != new_stride.size() && !new_size.empty() && !new_stride.empty()) {
            // If they don't match, use the smaller one's size
            size_t min_size = std::min(new_size.size(), new_stride.size());
            new_size.resize(min_size);
            new_stride.resize(min_size);
        }
        
        // Clone the tensor to avoid modifying the original
        torch::Tensor tensor_copy = input_tensor.clone();
        
        // Apply as_strided_ operation
        if (!new_size.empty() && !new_stride.empty()) {
            tensor_copy.as_strided_(new_size, new_stride, storage_offset);
        } else if (!new_size.empty()) {
            // If stride is empty but size is not, create default strides
            std::vector<int64_t> default_strides(new_size.size(), 1);
            for (int i = new_size.size() - 2; i >= 0; i--) {
                default_strides[i] = default_strides[i + 1] * new_size[i + 1];
            }
            tensor_copy.as_strided_(new_size, default_strides, storage_offset);
        } else {
            // If both are empty, use original tensor's size and default strides
            std::vector<int64_t> original_size;
            for (int i = 0; i < input_tensor.dim(); i++) {
                original_size.push_back(input_tensor.size(i));
            }
            
            std::vector<int64_t> default_strides(original_size.size(), 1);
            for (int i = original_size.size() - 2; i >= 0; i--) {
                default_strides[i] = default_strides[i + 1] * original_size[i + 1];
            }
            
            tensor_copy.as_strided_(original_size, default_strides, storage_offset);
        }
        
        // Try to access elements to ensure the tensor is valid
        if (tensor_copy.numel() > 0) {
            tensor_copy.item();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}