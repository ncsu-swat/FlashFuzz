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
        
        // Create input tensor from the fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply conj_physical operation
        torch::Tensor result = torch::conj_physical(input_tensor);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            result.item();
        }
        
        // Try some edge cases if we have more data
        if (offset + 1 < Size) {
            // Create a view of the tensor and apply conj_physical to it
            torch::Tensor view_tensor = input_tensor;
            if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
                view_tensor = input_tensor.slice(0, 0, input_tensor.size(0) - 1);
            }
            torch::Tensor view_result = torch::conj_physical(view_tensor);
            
            // Apply conj_physical in-place if possible
            if (input_tensor.is_complex()) {
                torch::Tensor clone_tensor = input_tensor.clone();
                clone_tensor.conj_physical_();
            }
        }
        
        // Test with non-complex tensors as well
        if (offset + 1 < Size) {
            // Create a real tensor (if the original was complex)
            torch::Tensor real_tensor;
            if (input_tensor.is_complex()) {
                real_tensor = torch::real(input_tensor);
            } else {
                real_tensor = input_tensor;
            }
            
            // Apply conj_physical to real tensor
            torch::Tensor real_result = torch::conj_physical(real_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
