#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least two tensors for vdot operation
        if (Size < 4) // Minimum bytes needed for basic tensor creation
            return 0;
            
        // Create first input tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if there's data left
        if (offset < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // vdot requires 1D tensors with same number of elements
            // We'll reshape tensors to 1D if needed
            int64_t numel1 = tensor1.numel();
            int64_t numel2 = tensor2.numel();
            
            // If either tensor has zero elements, we can't reshape
            if (numel1 > 0 && numel2 > 0) {
                // Reshape tensors to 1D if they're not already
                if (tensor1.dim() != 1) {
                    tensor1 = tensor1.reshape({numel1});
                }
                
                if (tensor2.dim() != 1) {
                    tensor2 = tensor2.reshape({numel2});
                }
                
                // If tensors have different number of elements, resize the second one
                // to match the first (either truncate or pad with zeros)
                if (numel1 != numel2) {
                    tensor2 = tensor2.reshape(-1);
                    if (numel2 > numel1) {
                        tensor2 = tensor2.slice(0, 0, numel1);
                    } else {
                        // Pad tensor2 with zeros
                        torch::Tensor padded = torch::zeros({numel1}, tensor2.options());
                        padded.slice(0, 0, numel2).copy_(tensor2);
                        tensor2 = padded;
                    }
                }
                
                // Apply vdot operation
                // If tensors have incompatible dtypes, this will throw an exception
                // which is caught by the outer try-catch
                torch::Tensor result = torch::vdot(tensor1, tensor2);
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