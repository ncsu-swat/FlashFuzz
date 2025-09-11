#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for tan operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply tan operation to the input tensor
        torch::Tensor output = torch::tan(input);
        
        // Try some variants of the operation
        if (offset + 1 < Size) {
            // Try in-place version if we have more data
            torch::Tensor input_copy = input.clone();
            input_copy.tan_();
            
            // Try creating a new tensor with specific options and apply tan
            torch::TensorOptions options = torch::TensorOptions()
                .dtype(input.dtype())
                .device(input.device());
            torch::Tensor new_tensor = torch::zeros_like(input, options);
            new_tensor.copy_(input);
            torch::Tensor output2 = torch::tan(new_tensor);
        }
        
        // Try with different tensor types if we have more data
        if (offset + 2 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Apply tan to this tensor too
            torch::Tensor output3 = torch::tan(input2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
