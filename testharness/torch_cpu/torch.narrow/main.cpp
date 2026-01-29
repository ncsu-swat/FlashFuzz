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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip 0-dim tensors as narrow requires at least 1 dimension
        if (input_tensor.dim() == 0) {
            return 0;
        }
        
        // Extract parameters for narrow operation
        // Get dim parameter (dimension along which to narrow)
        int64_t dim = 0;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t dim_byte = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
            // Map to valid dimension range [0, ndim-1]
            dim = std::abs(dim_byte) % input_tensor.dim();
        }
        
        // Get the size along the chosen dimension
        int64_t dim_size = input_tensor.size(dim);
        
        // Skip if dimension has size 0
        if (dim_size == 0) {
            return 0;
        }
        
        // Get start parameter (starting position)
        int64_t start = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&start, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Map start to valid range [0, dim_size-1]
            start = std::abs(start) % dim_size;
        }
        
        // Get length parameter (length of slice)
        int64_t length = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Map length to valid range [1, dim_size - start]
            int64_t max_length = dim_size - start;
            if (max_length > 0) {
                length = (std::abs(length) % max_length) + 1;
            } else {
                length = 1;
            }
        }
        
        // Apply narrow operation with valid parameters
        torch::Tensor result = torch::narrow(input_tensor, dim, start, length);
        
        // Verify result has expected shape
        (void)result.sizes();
        
        // Try the method version of narrow
        torch::Tensor result2 = input_tensor.narrow(dim, start, length);
        (void)result2.sizes();
        
        // Test with different dimensions if tensor has multiple dimensions
        if (input_tensor.dim() > 1) {
            int64_t alt_dim = (dim + 1) % input_tensor.dim();
            int64_t alt_dim_size = input_tensor.size(alt_dim);
            
            if (alt_dim_size > 0) {
                int64_t alt_start = start % alt_dim_size;
                int64_t alt_max_length = alt_dim_size - alt_start;
                int64_t alt_length = (alt_max_length > 0) ? ((length % alt_max_length) + 1) : 1;
                
                try {
                    torch::Tensor result3 = torch::narrow(input_tensor, alt_dim, alt_start, alt_length);
                    (void)result3.sizes();
                } catch (const std::exception &) {
                    // Silently ignore exceptions for alternative test
                }
            }
        }
        
        // Test narrow with tensor start (if available)
        try {
            torch::Tensor start_tensor = torch::tensor({start});
            torch::Tensor result4 = input_tensor.narrow(dim, start_tensor, length);
            (void)result4.sizes();
        } catch (const std::exception &) {
            // Silently ignore - tensor start may not be supported in all versions
        }
        
        // Test narrow_copy if available
        try {
            torch::Tensor result5 = torch::narrow_copy(input_tensor, dim, start, length);
            (void)result5.sizes();
        } catch (const std::exception &) {
            // Silently ignore - narrow_copy may not be available
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}