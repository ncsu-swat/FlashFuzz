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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.negative operation
        torch::Tensor result = torch::negative(input_tensor);
        
        // Try alternative API forms
        torch::Tensor result2 = -input_tensor;
        torch::Tensor result3 = input_tensor.neg();
        
        // Try in-place version if possible
        if (input_tensor.is_floating_point() || input_tensor.is_complex()) {
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.neg_();
        }
        
        // Try with different output tensor
        torch::Tensor output = torch::empty_like(input_tensor);
        torch::neg_out(output, input_tensor);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
