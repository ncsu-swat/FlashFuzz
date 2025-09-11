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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.cpu() operation to the tensor
        torch::Tensor cpu_tensor = tensor.cpu();
        
        // Try to access the tensor to ensure it's valid
        if (cpu_tensor.defined()) {
            auto sizes = cpu_tensor.sizes();
            auto dtype = cpu_tensor.dtype();
            
            // Try to perform some operations on the CPU tensor
            if (cpu_tensor.numel() > 0) {
                // Access first element to ensure tensor is accessible
                auto first_elem = cpu_tensor.flatten()[0];
                
                // Try a simple operation
                auto doubled = cpu_tensor * 2;
            }
        }
        
        // If there's more data, try creating another tensor with different device
        if (offset + 2 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Try with a tensor that might already be on CPU
            torch::Tensor cpu_tensor2 = tensor2.cpu();
            
            // Try with a tensor that we explicitly move to CPU (redundant but tests the API)
            torch::Tensor cpu_tensor3 = tensor2.to(torch::kCPU);
            
            // Verify both methods produce the same result
            if (cpu_tensor2.defined() && cpu_tensor3.defined()) {
                bool same_device = cpu_tensor2.device() == cpu_tensor3.device();
                bool same_dtype = cpu_tensor2.dtype() == cpu_tensor3.dtype();
                bool same_shape = cpu_tensor2.sizes() == cpu_tensor3.sizes();
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
