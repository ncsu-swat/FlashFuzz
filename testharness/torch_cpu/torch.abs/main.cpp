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
        
        // Apply torch.abs operation
        torch::Tensor result = torch::abs(input_tensor);
        
        // Try some variations of the abs operation
        if (offset + 1 < Size) {
            // Use out parameter version if we have more data
            torch::Tensor out_tensor = torch::empty_like(input_tensor);
            torch::abs_out(out_tensor, input_tensor);
            
            // Try the functional version
            torch::Tensor functional_result = torch::abs(input_tensor);
            
            // Try the method version
            torch::Tensor method_result = input_tensor.abs();
            
            // Try the in-place version
            torch::Tensor inplace_tensor = input_tensor.clone();
            inplace_tensor.abs_();
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            // Create another tensor with different properties
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply abs to this tensor too
            torch::Tensor another_result = torch::abs(another_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
