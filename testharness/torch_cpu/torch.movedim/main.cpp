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
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract source and destination dimensions
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Get tensor rank
        int64_t rank = input_tensor.dim();
        
        // If tensor is empty or scalar, try with simple parameters
        if (rank == 0) {
            try {
                torch::movedim(input_tensor, 0, 0);
            } catch (...) {
                // Expected to fail for scalar tensors
            }
            return 0;
        }
        
        // Parse source dimension
        int64_t source_dim;
        if (offset < Size) {
            source_dim = static_cast<int8_t>(Data[offset++]); // Use signed to allow negative dims
        } else {
            source_dim = 0;
        }
        
        // Parse destination dimension
        int64_t destination_dim;
        if (offset < Size) {
            destination_dim = static_cast<int8_t>(Data[offset++]); // Use signed to allow negative dims
        } else {
            destination_dim = 0;
        }
        
        // Test single dimension version
        try {
            torch::Tensor result1 = torch::movedim(input_tensor, source_dim, destination_dim);
            // Verify result has same number of elements
            (void)result1.numel();
        } catch (...) {
            // Expected to throw for invalid dimensions
        }
        
        // Test vector version if we have more data
        if (offset + 2 <= Size && rank > 1) {
            std::vector<int64_t> source_dims;
            std::vector<int64_t> destination_dims;
            
            // Parse number of dimensions to move (limited by tensor rank)
            uint8_t num_dims_to_move = Data[offset++] % rank + 1;
            
            // Parse source and destination dimensions
            for (uint8_t i = 0; i < num_dims_to_move && offset < Size; ++i) {
                int8_t src_dim = static_cast<int8_t>(Data[offset++]);
                source_dims.push_back(src_dim);
                
                if (offset < Size) {
                    int8_t dst_dim = static_cast<int8_t>(Data[offset++]);
                    destination_dims.push_back(dst_dim);
                } else {
                    destination_dims.push_back(0);
                }
            }
            
            // Ensure vectors have same size
            if (source_dims.size() != destination_dims.size()) {
                if (source_dims.size() < destination_dims.size()) {
                    destination_dims.resize(source_dims.size());
                } else {
                    source_dims.resize(destination_dims.size());
                }
            }
            
            // Test with vectors of dimensions
            try {
                torch::Tensor result2 = torch::movedim(input_tensor, source_dims, destination_dims);
                (void)result2.numel();
            } catch (...) {
                // Expected to throw for invalid dimensions
            }
        }
        
        // Test edge cases with empty vectors
        try {
            torch::Tensor result3 = torch::movedim(input_tensor, std::vector<int64_t>{}, std::vector<int64_t>{});
            (void)result3.numel();
        } catch (...) {
            // May throw depending on implementation
        }
        
        // Test with negative dimensions (valid in PyTorch)
        try {
            torch::Tensor result4 = torch::movedim(input_tensor, -1, 0);
            (void)result4.numel();
        } catch (...) {
            // May throw for invalid negative dimensions
        }
        
        // Test moving last dim to first position
        try {
            torch::Tensor result5 = torch::movedim(input_tensor, rank - 1, 0);
            (void)result5.numel();
        } catch (...) {
            // May throw
        }
        
        // Test with out-of-bounds dimensions (should throw)
        try {
            torch::Tensor result6 = torch::movedim(input_tensor, rank + 1, 0);
            (void)result6.numel();
        } catch (...) {
            // Expected to throw for out-of-bounds dimensions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}