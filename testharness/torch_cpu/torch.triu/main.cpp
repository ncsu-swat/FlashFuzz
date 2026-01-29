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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // triu works on at least 2D tensors, ensure we have proper dimensions
        if (input.dim() < 2) {
            // Reshape to 2D if 1D or 0D
            int64_t numel = input.numel();
            if (numel == 0) {
                return 0;
            }
            int64_t side = static_cast<int64_t>(std::sqrt(static_cast<double>(numel)));
            if (side < 1) side = 1;
            int64_t other = numel / side;
            if (other < 1) other = 1;
            input = input.flatten().narrow(0, 0, side * other).view({side, other});
        }
        
        // Parse diagonal parameter if we have more data
        int64_t diagonal = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&diagonal, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Clamp to reasonable range to avoid extreme values dominating
            diagonal = diagonal % 1000;
        }
        
        // Apply triu operation
        torch::Tensor result = torch::triu(input, diagonal);
        
        // Try inplace operation
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.triu_(diagonal);
        } catch (...) {
            // Ignore exceptions from inplace version
        }
        
        // Try with different diagonal values if we have more data
        if (offset + sizeof(int8_t) <= Size) {
            int8_t diagonal2_small;
            std::memcpy(&diagonal2_small, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            
            try {
                torch::Tensor result2 = torch::triu(input, static_cast<int64_t>(diagonal2_small));
            } catch (...) {
                // Ignore exceptions from different diagonal
            }
        }
        
        // Test boundary cases with diagonal relative to tensor dimensions
        int64_t rows = input.size(-2);
        int64_t cols = input.size(-1);
        
        try {
            // Diagonal at the edge
            torch::Tensor result_edge_pos = torch::triu(input, cols);
        } catch (...) {
            // Ignore exceptions
        }
        
        try {
            // Diagonal below main
            torch::Tensor result_edge_neg = torch::triu(input, -rows);
        } catch (...) {
            // Ignore exceptions
        }
        
        // Test with 3D tensor (batched operation) if input was 2D
        if (input.dim() == 2 && input.numel() > 0) {
            try {
                torch::Tensor batched = input.unsqueeze(0).expand({2, -1, -1}).contiguous();
                torch::Tensor batched_result = torch::triu(batched, diagonal);
            } catch (...) {
                // Ignore exceptions from batched version
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