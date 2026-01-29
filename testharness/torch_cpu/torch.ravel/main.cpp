#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.ravel operation
        // torch.ravel returns a contiguous flattened tensor
        torch::Tensor result = input_tensor.ravel();
        
        // Verify the result has the expected properties
        // The result should be 1D and have the same number of elements as the input
        // Use assert-style check instead of throwing
        assert(result.dim() == 1);
        assert(result.numel() == input_tensor.numel());
        
        // Try to access elements to ensure the tensor is valid
        if (result.numel() > 0) {
            result[0].item();
        }
        
        // Try alternative ways to call ravel
        torch::Tensor result2 = torch::ravel(input_tensor);
        
        // Try with a view of the tensor (may fail for non-contiguous tensors)
        try {
            if (input_tensor.numel() > 0) {
                torch::Tensor view_tensor = input_tensor.view({-1});
                torch::Tensor result3 = view_tensor.ravel();
            }
        } catch (...) {
            // view may fail for non-contiguous tensors, that's expected
        }
        
        // Try with a non-contiguous tensor if possible
        try {
            if (input_tensor.dim() > 1 && input_tensor.size(0) > 1 && input_tensor.size(1) > 1) {
                torch::Tensor transposed = input_tensor.transpose(0, 1);
                torch::Tensor result4 = transposed.ravel();
                // Verify transposed ravel also produces correct result
                assert(result4.dim() == 1);
                assert(result4.numel() == transposed.numel());
            }
        } catch (...) {
            // transpose may fail for certain tensor configurations
        }
        
        // Try with a zero-sized tensor
        try {
            std::vector<int64_t> empty_shape = {0};
            torch::Tensor empty_tensor = torch::empty(empty_shape, input_tensor.options());
            torch::Tensor result5 = empty_tensor.ravel();
            assert(result5.dim() == 1);
            assert(result5.numel() == 0);
        } catch (...) {
            // Empty tensor operations may have edge cases
        }
        
        // Try with different tensor types for better coverage
        try {
            torch::Tensor float_tensor = input_tensor.to(torch::kFloat);
            torch::Tensor result6 = float_tensor.ravel();
        } catch (...) {
            // Type conversion may fail
        }
        
        // Try with a cloned tensor to test on contiguous memory
        try {
            torch::Tensor cloned = input_tensor.clone();
            torch::Tensor result7 = cloned.ravel();
        } catch (...) {
            // Clone may fail in edge cases
        }
        
        // Try ravel on a slice if tensor has enough elements
        try {
            if (input_tensor.dim() >= 1 && input_tensor.size(0) > 1) {
                torch::Tensor slice = input_tensor.slice(0, 0, input_tensor.size(0) / 2 + 1);
                torch::Tensor result8 = slice.ravel();
            }
        } catch (...) {
            // Slicing may fail for certain shapes
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}