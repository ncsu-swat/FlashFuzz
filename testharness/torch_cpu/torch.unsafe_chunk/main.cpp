#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Check if tensor is valid and non-empty
        if (!input_tensor.defined() || input_tensor.numel() == 0) {
            return 0;
        }
        
        // Extract parameters for unsafe_chunk from remaining data
        int64_t chunks = 2; // Default value
        int64_t dim = 0;    // Default value
        
        // Get chunks parameter if we have enough data
        if (offset + sizeof(uint8_t) <= Size) {
            // Use a single byte to get reasonable chunk values (1-256)
            chunks = static_cast<int64_t>(Data[offset]) + 1; // Ensure chunks >= 1
            offset += sizeof(uint8_t);
        }
        
        // Get dimension parameter if we have enough data
        if (offset + sizeof(uint8_t) <= Size) {
            // Use a single byte and constrain to valid dimension range
            int64_t ndim = input_tensor.dim();
            if (ndim > 0) {
                int8_t raw_dim = static_cast<int8_t>(Data[offset]);
                // Map to valid dimension range [-ndim, ndim-1]
                dim = raw_dim % ndim;
            }
            offset += sizeof(uint8_t);
        }
        
        // Ensure dim is valid for the tensor
        int64_t ndim = input_tensor.dim();
        if (ndim == 0) {
            // Scalar tensor, can't chunk
            return 0;
        }
        
        // Normalize negative dim
        if (dim < 0) {
            dim = dim + ndim;
        }
        
        // Clamp dim to valid range
        if (dim < 0 || dim >= ndim) {
            dim = 0;
        }
        
        // Apply the unsafe_chunk operation
        // unsafe_chunk doesn't check if split is even, so we test it
        std::vector<torch::Tensor> chunks_result;
        try {
            chunks_result = torch::unsafe_chunk(input_tensor, chunks, dim);
        } catch (const c10::Error&) {
            // Expected failure for invalid parameters (silently catch)
            return 0;
        }
        
        // Perform operations on the result to ensure coverage
        if (!chunks_result.empty()) {
            for (auto& chunk : chunks_result) {
                auto sizes = chunk.sizes();
                auto numel = chunk.numel();
                auto dtype = chunk.dtype();
                
                // Perform operations to ensure tensor is valid
                if (numel > 0 && chunk.is_floating_point()) {
                    torch::Tensor sum = chunk.sum();
                    torch::Tensor mean_val = chunk.mean();
                }
                
                // Test contiguity
                bool is_contiguous = chunk.is_contiguous();
                
                // Test clone to exercise memory
                if (numel > 0 && numel < 1000) {
                    torch::Tensor cloned = chunk.clone();
                }
            }
        }
        
        // Also test with different chunk counts to improve coverage
        if (offset + sizeof(uint8_t) <= Size) {
            int64_t alt_chunks = (static_cast<int64_t>(Data[offset]) % 8) + 1;
            offset += sizeof(uint8_t);
            
            try {
                auto alt_result = torch::unsafe_chunk(input_tensor, alt_chunks, dim);
                // Access results
                for (const auto& t : alt_result) {
                    volatile auto n = t.numel();
                    (void)n;
                }
            } catch (const c10::Error&) {
                // Silently catch expected failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}