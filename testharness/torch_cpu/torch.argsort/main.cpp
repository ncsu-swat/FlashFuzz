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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for argsort
        int64_t dim = 0;
        bool descending = false;
        bool stable = false;
        
        // Parse dim parameter if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_dim;
            std::memcpy(&raw_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor is not empty, modulo by number of dimensions to get valid dim
            if (input_tensor.dim() > 0) {
                dim = ((raw_dim % input_tensor.dim()) + input_tensor.dim()) % input_tensor.dim();
            }
        }
        
        // Parse descending parameter if we have more data
        if (offset < Size) {
            descending = Data[offset++] & 0x1;
        }
        
        // Parse stable parameter if we have more data
        if (offset < Size) {
            stable = Data[offset++] & 0x1;
        }
        
        // Apply argsort operation
        torch::Tensor result;
        
        // Try different variants of argsort
        if (offset % 3 == 0) {
            // Variant 1: argsort with dim, descending, stable
            result = torch::argsort(input_tensor, dim, descending, stable);
        } 
        else if (offset % 3 == 1) {
            // Variant 2: argsort with dim, descending
            result = torch::argsort(input_tensor, dim, descending);
        }
        else {
            // Variant 3: argsort with dim only
            result = torch::argsort(input_tensor, dim);
        }
        
        // Verify result is not empty
        if (result.numel() != input_tensor.numel()) {
            throw std::runtime_error("Result tensor has different number of elements than input tensor");
        }
        
        // Try to use the result to ensure it's not optimized away
        auto sum = result.sum().item<double>();
        if (std::isnan(sum) || std::isinf(sum)) {
            throw std::runtime_error("Result contains NaN or Inf values");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
