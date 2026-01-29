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
        
        // Need at least a few bytes for tensor creation and dimension indices
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get tensor rank
        int64_t rank = input_tensor.dim();
        
        // If we've consumed all data or tensor is scalar, return
        if (offset + 2 > Size || rank < 2) {
            return 0;
        }
        
        // Extract dimension indices for swapping
        int64_t dim1 = static_cast<int8_t>(Data[offset++]); // Use signed int8_t to allow negative indices
        int64_t dim2 = static_cast<int8_t>(Data[offset++]);
        
        // Apply swapdims operation - wrap in inner try-catch for expected dimension errors
        try {
            torch::Tensor result = torch::swapdims(input_tensor, dim1, dim2);
            
            // Verify result has same number of elements
            if (result.numel() != input_tensor.numel()) {
                throw std::runtime_error("Result tensor has different number of elements");
            }
        } catch (const c10::Error&) {
            // Expected for invalid dimension indices - silently ignore
        }
        
        // Try method variant of the API
        try {
            torch::Tensor result = input_tensor.swapdims(dim1, dim2);
        } catch (const c10::Error&) {
            // Expected for invalid dimension indices - silently ignore
        }
        
        // Try with different dimensions if we have more data
        if (offset + 2 <= Size && rank >= 2) {
            int64_t dim1_new = static_cast<int8_t>(Data[offset++]);
            int64_t dim2_new = static_cast<int8_t>(Data[offset++]);
            
            try {
                torch::Tensor result = torch::swapdims(input_tensor, dim1_new, dim2_new);
            } catch (const c10::Error&) {
                // Expected for invalid dimension indices - silently ignore
            }
            
            // Update dims for transpose test
            dim1 = dim1_new;
            dim2 = dim2_new;
        }
        
        // Try the alias transpose (which is essentially the same as swapdims)
        if (rank >= 2) {
            try {
                torch::Tensor result = torch::transpose(input_tensor, dim1, dim2);
            } catch (const c10::Error&) {
                // Expected for invalid dimension indices - silently ignore
            }
            
            try {
                torch::Tensor result = input_tensor.transpose(dim1, dim2);
            } catch (const c10::Error&) {
                // Expected for invalid dimension indices - silently ignore
            }
        }
        
        // Test with contiguous tensor variant
        try {
            torch::Tensor contiguous_input = input_tensor.contiguous();
            torch::Tensor result = torch::swapdims(contiguous_input, dim1, dim2);
        } catch (const c10::Error&) {
            // Expected for invalid dimension indices - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}