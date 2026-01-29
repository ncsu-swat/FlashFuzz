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
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor for tan operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply tan operation to the input tensor
        torch::Tensor output = torch::tan(input);
        
        // Try in-place version
        torch::Tensor input_copy = input.clone();
        input_copy.tan_();
        
        // Try with output tensor (out= variant)
        torch::Tensor out_tensor = torch::empty_like(input);
        torch::tan_out(out_tensor, input);
        
        // Try creating a new tensor with specific options and apply tan
        torch::TensorOptions options = torch::TensorOptions()
            .dtype(input.dtype())
            .device(input.device());
        torch::Tensor new_tensor = torch::zeros_like(input, options);
        new_tensor.copy_(input);
        torch::Tensor output2 = torch::tan(new_tensor);
        
        // Try with different tensor types if we have more data
        if (offset < Size) {
            try {
                // Create another tensor with potentially different properties
                torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Apply tan to this tensor too
                torch::Tensor output3 = torch::tan(input2);
                
                // Try in-place on second tensor
                input2.tan_();
            }
            catch (...) {
                // Silently ignore failures on secondary tensor operations
            }
        }
        
        // Test with specific dtypes that tan supports
        try {
            torch::Tensor float_input = input.to(torch::kFloat32);
            torch::Tensor float_output = torch::tan(float_input);
        }
        catch (...) {
            // Silently ignore dtype conversion failures
        }
        
        try {
            torch::Tensor double_input = input.to(torch::kFloat64);
            torch::Tensor double_output = torch::tan(double_input);
        }
        catch (...) {
            // Silently ignore dtype conversion failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}