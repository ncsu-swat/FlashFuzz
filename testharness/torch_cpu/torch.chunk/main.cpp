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
        
        // Need at least a few bytes to create a tensor and parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for chunk operation
        // Need at least 2 more bytes for chunks and dim
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Get number of chunks (ensure at least 1)
        int64_t chunks = static_cast<int64_t>(Data[offset++]) + 1;
        
        // Get dimension to split along
        int64_t dim;
        if (input_tensor.dim() == 0) {
            // For scalar tensors, use dim=0
            dim = 0;
        } else {
            // For non-scalar tensors, allow negative dimensions
            dim = static_cast<int8_t>(Data[offset++]);
        }
        
        // Apply torch.chunk operation
        std::vector<torch::Tensor> result = torch::chunk(input_tensor, chunks, dim);
        
        // Verify the result by checking properties
        for (const auto& chunk : result) {
            // Access tensor properties to ensure they're valid
            auto sizes = chunk.sizes();
            auto dtype = chunk.dtype();
            auto numel = chunk.numel();
            
            // Perform a simple operation on each chunk to ensure it's usable
            if (numel > 0) {
                auto sum = torch::sum(chunk);
            }
        }
        
        // Test with different parameters if we have more data
        if (offset + 2 <= Size) {
            // Try with different chunks value
            int64_t chunks2 = static_cast<int64_t>(Data[offset++]) + 1;
            
            // Try with different dimension
            int64_t dim2;
            if (input_tensor.dim() == 0) {
                dim2 = 0;
            } else {
                dim2 = static_cast<int8_t>(Data[offset++]);
            }
            
            // Apply torch.chunk with different parameters
            std::vector<torch::Tensor> result2 = torch::chunk(input_tensor, chunks2, dim2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
