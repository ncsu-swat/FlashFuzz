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
        
        // Create a tensor with device specification
        torch::Tensor tensor;
        
        // Determine if we should create a CPU or CUDA tensor
        bool use_cuda = false;
        if (Size > 0) {
            use_cuda = (Data[0] % 2 == 1) && torch::cuda::is_available();
            offset++;
        }
        
        // Create the tensor
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Move tensor to CUDA if needed
            if (use_cuda) {
                tensor = tensor.cuda();
            }
            
            // Test get_device functionality
            int device = tensor.get_device();
            
            // For CPU tensors, get_device() returns -1
            // For CUDA tensors, get_device() returns the device index (0, 1, etc.)
            
            // Test additional device-related operations
            bool is_cuda = tensor.is_cuda();
            bool is_cpu = tensor.is_cpu();
            
            // Try to get device for a view of the tensor
            torch::Tensor view_tensor = tensor.view({-1});
            int view_device = view_tensor.get_device();
            
            // Try to get device for a slice of the tensor if possible
            if (tensor.dim() > 0 && tensor.size(0) > 1) {
                torch::Tensor slice_tensor = tensor.slice(0, 0, 1);
                int slice_device = slice_tensor.get_device();
            }
            
            // Try to get device for a transposed tensor if possible
            if (tensor.dim() >= 2) {
                torch::Tensor transposed = tensor.transpose(0, 1);
                int transposed_device = transposed.get_device();
            }
            
            // Try to get device for a tensor after in-place operations
            if (tensor.numel() > 0 && tensor.is_floating_point()) {
                torch::Tensor clone_tensor = tensor.clone();
                clone_tensor.mul_(2.0);
                int clone_device = clone_tensor.get_device();
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