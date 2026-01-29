#include "fuzzer_utils.h"
#include <iostream>

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
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Flatten input to get total number of elements
        int64_t numel = input_tensor.numel();
        if (numel == 0) {
            return 0; // Can't take from empty tensor
        }
        
        // Create indices tensor
        torch::Tensor indices_tensor;
        if (offset + 4 < Size) {
            indices_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Convert to Int64 for indices
            if (!indices_tensor.is_floating_point()) {
                indices_tensor = indices_tensor.to(torch::kInt64);
            } else {
                indices_tensor = indices_tensor.to(torch::kInt64);
            }
            
            // Clamp indices to valid range [-numel, numel-1]
            // torch::take treats the input as 1D and uses flattened indices
            // Negative indices wrap around
            indices_tensor = indices_tensor.remainder(numel);
        } else {
            // Create a simple indices tensor with valid indices
            int64_t idx0 = 0;
            int64_t idx1 = numel > 1 ? 1 : 0;
            int64_t idx2 = numel > 2 ? -1 : 0; // -1 wraps to last element
            indices_tensor = torch::tensor({idx0, idx1, idx2}, torch::kInt64);
        }
        
        // Inner try-catch for expected exceptions (out of bounds, etc.)
        try
        {
            // torch::take - Returns a new tensor with elements of input at given indices
            // The input tensor is treated as if it were viewed as a 1-D tensor
            torch::Tensor result = torch::take(input_tensor, indices_tensor);
            
            // Verify result shape matches indices shape
            // torch::take returns tensor with same shape as indices
            
            // Perform operations on result to ensure it's computed
            auto sum = result.sum();
            
            // Prevent compiler optimization
            if (sum.item<float>() == -12345.6789f) {
                std::cerr << "Unlikely sum value encountered";
            }
            
            // Test with contiguous input
            if (offset < Size && Data[offset % Size] % 2 == 0) {
                torch::Tensor contiguous_input = input_tensor.contiguous();
                torch::Tensor result2 = torch::take(contiguous_input, indices_tensor);
                auto sum2 = result2.sum();
                (void)sum2;
            }
            
            // Test with different tensor strides (transposed tensor)
            if (input_tensor.dim() >= 2 && offset + 1 < Size && Data[(offset + 1) % Size] % 3 == 0) {
                torch::Tensor transposed = input_tensor.transpose(0, 1);
                torch::Tensor result3 = torch::take(transposed, indices_tensor);
                auto sum3 = result3.sum();
                (void)sum3;
            }
        }
        catch (const c10::Error &e)
        {
            // Expected errors (index out of bounds, etc.) - silently ignore
        }
        catch (const std::runtime_error &e)
        {
            // Expected runtime errors - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}