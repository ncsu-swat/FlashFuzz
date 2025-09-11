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
        
        // Apply torch.sin operation
        torch::Tensor result = torch::sin(input_tensor);
        
        // Try some variations if we have more data
        if (offset + 1 < Size) {
            // Try in-place version
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.sin_();
            
            // Try with out parameter
            torch::Tensor out_tensor = torch::empty_like(input_tensor);
            torch::sin_out(out_tensor, input_tensor);
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            uint8_t option_selector = Data[offset++];
            
            // Try with non-contiguous tensor
            if (option_selector % 4 == 0 && input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
                torch::Tensor non_contiguous = input_tensor.transpose(0, input_tensor.dim() - 1);
                torch::Tensor result_non_contiguous = torch::sin(non_contiguous);
            }
            
            // Try with different device if available
            if (option_selector % 4 == 1 && torch::cuda::is_available()) {
                torch::Tensor cuda_tensor = input_tensor.cuda();
                torch::Tensor cuda_result = torch::sin(cuda_tensor);
            }
            
            // Try with requires_grad
            if (option_selector % 4 == 2) {
                // Only floating point types support autograd
                if (input_tensor.is_floating_point()) {
                    torch::Tensor grad_tensor = input_tensor.clone().detach().requires_grad_(true);
                    torch::Tensor grad_result = torch::sin(grad_tensor);
                    
                    // Try backward if tensor is scalar or has small number of elements
                    if (input_tensor.numel() < 10) {
                        grad_result.backward();
                    }
                }
            }
            
            // Try with different dtype if possible
            if (option_selector % 4 == 3) {
                torch::ScalarType target_dtype = fuzzer_utils::parseDataType(Data[offset++]);
                // Only attempt conversion if it makes sense
                if (torch::can_cast(input_tensor.scalar_type(), target_dtype)) {
                    torch::Tensor converted = input_tensor.to(target_dtype);
                    torch::Tensor converted_result = torch::sin(converted);
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
