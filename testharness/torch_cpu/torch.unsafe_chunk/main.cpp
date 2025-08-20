#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for unsafe_chunk from remaining data
        int64_t chunks = 2; // Default value
        int64_t dim = 0;    // Default value
        
        // Get chunks parameter if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&chunks, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure chunks is not zero (would cause division by zero)
            if (chunks == 0) {
                chunks = 1;
            }
        }
        
        // Get dimension parameter if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply the unsafe_chunk operation
        // Note: We're not adding defensive checks to allow testing edge cases
        std::vector<torch::Tensor> chunks_result = torch::unsafe_chunk(input_tensor, chunks, dim);
        
        // Perform some basic operations on the result to ensure it's used
        if (!chunks_result.empty()) {
            for (auto& chunk : chunks_result) {
                auto sizes = chunk.sizes();
                auto numel = chunk.numel();
                auto dtype = chunk.dtype();
                
                // Perform a simple operation to ensure tensor is valid
                if (numel > 0) {
                    torch::Tensor sum = chunk.sum();
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