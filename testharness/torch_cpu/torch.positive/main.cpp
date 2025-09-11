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
        
        // Apply torch.positive operation
        torch::Tensor result = torch::positive(input_tensor);
        
        // Try to access the result to ensure computation is performed
        if (result.defined()) {
            auto sizes = result.sizes();
            auto dtype = result.dtype();
            
            // Force evaluation by accessing an element if tensor is not empty
            if (result.numel() > 0) {
                result.item();
            }
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor another_result = torch::positive(another_input);
            
            // Force evaluation
            if (another_result.defined() && another_result.numel() > 0) {
                another_result.item();
            }
        }
        
        // Test with non-contiguous tensor if we have enough data
        if (offset + 2 < Size && input_tensor.dim() > 0 && input_tensor.numel() > 1) {
            // Create a non-contiguous view if possible
            torch::Tensor non_contiguous;
            if (input_tensor.dim() > 1 && input_tensor.size(0) > 1) {
                non_contiguous = input_tensor.transpose(0, input_tensor.dim() - 1);
            } else {
                // For 1D tensors or tensors with first dim = 1, try another approach
                non_contiguous = input_tensor.expand({2, -1});
            }
            
            // Apply positive to non-contiguous tensor
            if (!non_contiguous.is_contiguous()) {
                torch::Tensor non_contiguous_result = torch::positive(non_contiguous);
                
                // Force evaluation
                if (non_contiguous_result.defined() && non_contiguous_result.numel() > 0) {
                    non_contiguous_result.item();
                }
            }
        }
        
        // Test with different dtypes if we have more data
        if (offset + 2 < Size) {
            // Try to create a tensor with a different dtype
            torch::Tensor typed_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply positive
            torch::Tensor typed_result = torch::positive(typed_tensor);
            
            // Force evaluation
            if (typed_result.defined() && typed_result.numel() > 0) {
                typed_result.item();
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
