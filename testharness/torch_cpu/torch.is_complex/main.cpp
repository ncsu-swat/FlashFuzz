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
        
        // Skip if there's not enough data to create a tensor
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the is_complex operation
        bool result = torch::is_complex(tensor);
        
        // Use the result to prevent compiler optimization
        if (result) {
            // Perform some operation to ensure the result is used
            torch::Tensor dummy = tensor.conj();
        }
        
        // Try with a view of the tensor if possible
        if (tensor.dim() > 0 && tensor.numel() > 0) {
            torch::Tensor view = tensor.view({-1});
            bool view_result = torch::is_complex(view);
            
            // Use the result
            if (view_result) {
                torch::Tensor dummy = view.conj();
            }
        }
        
        // Try with a slice of the tensor if possible
        if (tensor.dim() > 0 && tensor.size(0) > 1) {
            torch::Tensor slice = tensor.slice(0, 0, tensor.size(0) / 2);
            bool slice_result = torch::is_complex(slice);
            
            // Use the result
            if (slice_result) {
                torch::Tensor dummy = slice.conj();
            }
        }
        
        // Try with a transposed tensor if possible
        if (tensor.dim() >= 2) {
            torch::Tensor transposed = tensor.transpose(0, 1);
            bool transposed_result = torch::is_complex(transposed);
            
            // Use the result
            if (transposed_result) {
                torch::Tensor dummy = transposed.conj();
            }
        }
        
        // Try with a contiguous tensor
        torch::Tensor contiguous = tensor.contiguous();
        bool contiguous_result = torch::is_complex(contiguous);
        
        // Use the result
        if (contiguous_result) {
            torch::Tensor dummy = contiguous.conj();
        }
        
        // Try with a clone of the tensor
        torch::Tensor clone = tensor.clone();
        bool clone_result = torch::is_complex(clone);
        
        // Use the result
        if (clone_result) {
            torch::Tensor dummy = clone.conj();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
