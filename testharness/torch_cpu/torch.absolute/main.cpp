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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.absolute operation (torch.absolute is an alias for torch.abs)
        torch::Tensor result = torch::abs(input_tensor);
        
        // Try method syntax
        torch::Tensor result2 = input_tensor.abs();
        
        // Try in-place version on a clone (works on floating point, complex, and signed integers)
        try {
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.abs_();
        } catch (...) {
            // Some dtypes may not support in-place abs, silently ignore
        }
        
        // Try with out parameter
        try {
            torch::Tensor out_tensor = torch::empty_like(input_tensor);
            torch::abs_out(out_tensor, input_tensor);
        } catch (...) {
            // Silently catch if out parameter version fails
        }
        
        // Test with different tensor configurations if we have more data
        if (offset + 4 < Size) {
            // Create another tensor with remaining data
            torch::Tensor input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor result3 = torch::abs(input_tensor2);
        }
        
        // Test with scalar (0-dimensional tensor)
        if (Size >= 4) {
            float scalar_val;
            memcpy(&scalar_val, Data, sizeof(float));
            torch::Tensor scalar_tensor = torch::tensor(scalar_val);
            torch::Tensor scalar_result = torch::abs(scalar_tensor);
        }
        
        // Test with negative values explicitly
        if (input_tensor.is_floating_point()) {
            torch::Tensor neg_tensor = -input_tensor;
            torch::Tensor neg_result = torch::abs(neg_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}