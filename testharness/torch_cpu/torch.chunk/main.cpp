#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }
    
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor and parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Need at least 2 more bytes for chunks and dim
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Get number of chunks (ensure at least 1, cap at reasonable value)
        int64_t chunks = (static_cast<int64_t>(Data[offset++]) % 16) + 1;
        
        // Get dimension to split along - must be valid for the tensor
        int64_t dim = 0;
        int64_t ndim = input_tensor.dim();
        if (ndim > 0) {
            // Map byte to valid dimension range [-ndim, ndim-1]
            int64_t dim_range = ndim * 2;
            dim = (static_cast<int64_t>(Data[offset++]) % dim_range) - ndim;
        } else {
            offset++; // consume the byte anyway
        }
        
        // Apply torch.chunk operation - wrap in inner try-catch for expected failures
        std::vector<torch::Tensor> result;
        try {
            result = torch::chunk(input_tensor, chunks, dim);
        } catch (const c10::Error&) {
            // Expected for invalid dimension combinations, silently continue
            return 0;
        }
        
        // Verify the result by checking properties
        for (const auto& chunk : result) {
            // Access tensor properties to ensure they're valid
            auto sizes = chunk.sizes();
            auto dtype = chunk.dtype();
            auto numel = chunk.numel();
            
            // Perform a simple operation on each chunk to ensure it's usable
            if (numel > 0) {
                auto sum = torch::sum(chunk);
                (void)sum; // Prevent unused variable warning
            }
        }
        
        // Test with different parameters if we have more data
        if (offset + 2 <= Size) {
            // Try with different chunks value
            int64_t chunks2 = (static_cast<int64_t>(Data[offset++]) % 16) + 1;
            
            // Try with different dimension
            int64_t dim2 = 0;
            if (ndim > 0) {
                int64_t dim_range = ndim * 2;
                dim2 = (static_cast<int64_t>(Data[offset++]) % dim_range) - ndim;
            } else {
                offset++;
            }
            
            // Apply torch.chunk with different parameters
            try {
                std::vector<torch::Tensor> result2 = torch::chunk(input_tensor, chunks2, dim2);
                (void)result2; // Prevent unused variable warning
            } catch (const c10::Error&) {
                // Expected for invalid dimension combinations, silently continue
            }
        }
        
        // Test tensor_split as related API (similar to chunk but different interface)
        if (offset + 1 <= Size) {
            int64_t sections = (static_cast<int64_t>(Data[offset++]) % 8) + 1;
            try {
                auto split_result = torch::tensor_split(input_tensor, sections);
                (void)split_result;
            } catch (const c10::Error&) {
                // Expected for some inputs
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}