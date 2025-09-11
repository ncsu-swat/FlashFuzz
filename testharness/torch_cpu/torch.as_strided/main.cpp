#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse parameters for as_strided
        // We need at least a few more bytes for the parameters
        if (Size - offset < 4) {
            return 0;
        }
        
        // Parse size (shape) for as_strided
        uint8_t size_rank = Data[offset++] % 5; // Limit rank to 0-4
        std::vector<int64_t> size_vec = fuzzer_utils::parseShape(Data, offset, Size, size_rank);
        
        // Parse strides for as_strided
        uint8_t stride_rank = size_rank; // Strides should match size dimensions
        std::vector<int64_t> stride_vec;
        if (stride_rank > 0) {
            stride_vec = fuzzer_utils::parseShape(Data, offset, Size, stride_rank);
        }
        
        // Parse storage_offset
        int64_t storage_offset = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&storage_offset, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make storage_offset non-negative to avoid definite errors
            // but still allow for interesting edge cases
            storage_offset = std::abs(storage_offset) % 100;
        }
        
        // Apply as_strided operation
        torch::Tensor result;
        
        // Handle scalar tensor case (rank 0)
        if (size_rank == 0) {
            // For scalar tensors, use empty vectors
            result = input_tensor.as_strided({}, {}, storage_offset);
        } else {
            result = input_tensor.as_strided(size_vec, stride_vec, storage_offset);
        }
        
        // Basic validation - just access some elements to ensure it's valid
        if (result.numel() > 0) {
            auto item = result.item();
        }
        
        // Try some operations on the result tensor
        if (result.numel() > 0) {
            auto sum = result.sum();
            auto mean = result.mean();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
