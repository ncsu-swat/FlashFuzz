#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.resolve_neg operation
        torch::Tensor result = torch::resolve_neg(input_tensor);
        
        // Try to access the result to ensure computation is performed
        if (result.defined()) {
            auto sizes = result.sizes();
            auto dtype = result.dtype();
            
            // Force evaluation of the tensor
            if (result.numel() > 0) {
                result.item();
            }
        }
        
        // Try with a negative tensor if we have more data
        if (offset + 1 < Size) {
            torch::Tensor neg_tensor = -input_tensor;
            torch::Tensor neg_result = torch::resolve_neg(neg_tensor);
            
            if (neg_result.defined() && neg_result.numel() > 0) {
                neg_result.item();
            }
        }
        
        // Try with a zero tensor if we have more data
        if (offset + 1 < Size) {
            torch::Tensor zero_tensor = torch::zeros_like(input_tensor);
            torch::Tensor zero_result = torch::resolve_neg(zero_tensor);
            
            if (zero_result.defined() && zero_result.numel() > 0) {
                zero_result.item();
            }
        }
        
        // Try with a scalar tensor
        if (offset + 1 < Size) {
            torch::Tensor scalar_tensor = torch::tensor(Data[offset] % 256 - 128);
            torch::Tensor scalar_result = torch::resolve_neg(scalar_tensor);
            
            if (scalar_result.defined()) {
                scalar_result.item();
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