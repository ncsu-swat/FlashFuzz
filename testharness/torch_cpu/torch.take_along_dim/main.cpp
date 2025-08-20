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
        
        // Create indices tensor
        torch::Tensor indices_tensor;
        if (offset < Size) {
            indices_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure indices tensor has the same shape as input tensor or is broadcastable
            // Convert indices to long type as required by take_along_dim
            indices_tensor = indices_tensor.to(torch::kInt64);
        } else {
            // If we don't have enough data for a second tensor, create a simple indices tensor
            if (input_tensor.dim() > 0) {
                // Create indices with same shape as input
                indices_tensor = torch::zeros_like(input_tensor, torch::kInt64);
            } else {
                // For scalar tensors, create a simple scalar index
                indices_tensor = torch::zeros({}, torch::kInt64);
            }
        }
        
        // Get a dimension value to use with take_along_dim
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply take_along_dim operation
        // If input is a scalar (0-dim tensor), dim parameter is ignored
        if (input_tensor.dim() > 0) {
            // Allow negative dimensions to test edge cases
            torch::Tensor result = torch::take_along_dim(input_tensor, indices_tensor, dim);
        } else {
            // For scalar tensors, dim is ignored
            torch::Tensor result = torch::take_along_dim(input_tensor, indices_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}