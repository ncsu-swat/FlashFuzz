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
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor with various data types
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply is_floating_point operation
        bool is_float = torch::is_floating_point(tensor);
        
        // Try to force evaluation to ensure operation completes
        if (is_float) {
            // Do something trivial with the result to prevent optimization
            torch::Tensor dummy = tensor + 1.0;
        }
        
        // Try with a view of the tensor if possible
        if (tensor.dim() > 0 && tensor.numel() > 0) {
            torch::Tensor view_tensor = tensor.view({-1});
            bool is_view_float = torch::is_floating_point(view_tensor);
        }
        
        // Try with a slice of the tensor if possible
        if (tensor.dim() > 0 && tensor.numel() > 1) {
            torch::Tensor slice_tensor = tensor.slice(0, 0, tensor.size(0) / 2 + 1);
            bool is_slice_float = torch::is_floating_point(slice_tensor);
        }
        
        // Try with a transposed tensor if possible
        if (tensor.dim() >= 2) {
            torch::Tensor transposed = tensor.transpose(0, tensor.dim() - 1);
            bool is_transposed_float = torch::is_floating_point(transposed);
        }
        
        // Try with a contiguous tensor
        torch::Tensor contiguous = tensor.contiguous();
        bool is_contiguous_float = torch::is_floating_point(contiguous);
        
        // Try with a clone
        torch::Tensor clone = tensor.clone();
        bool is_clone_float = torch::is_floating_point(clone);
        
        // Try with a detached tensor
        torch::Tensor detached = tensor.detach();
        bool is_detached_float = torch::is_floating_point(detached);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
