#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.cpu() operation to the tensor
        // Since we're running on CPU, this should be a no-op but still tests the API
        torch::Tensor cpu_tensor = tensor.cpu();
        
        // Try to access the tensor to ensure it's valid
        if (cpu_tensor.defined()) {
            auto sizes = cpu_tensor.sizes();
            auto dtype = cpu_tensor.dtype();
            
            // Verify the tensor is on CPU device
            bool is_cpu = cpu_tensor.device().is_cpu();
            
            // Try to perform some operations on the CPU tensor
            if (cpu_tensor.numel() > 0) {
                // Access first element to ensure tensor is accessible
                auto first_elem = cpu_tensor.flatten()[0];
                
                // Try a simple operation
                auto doubled = cpu_tensor * 2;
            }
        }
        
        // If there's more data, try creating another tensor
        if (offset < Size) {
            size_t offset2 = 0;
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset2);
            
            // Try with a tensor - call cpu() on it
            torch::Tensor cpu_tensor2 = tensor2.cpu();
            
            // Try with to(kCPU) for comparison - both should work identically
            torch::Tensor cpu_tensor3 = tensor2.to(torch::kCPU);
            
            // Verify both methods produce tensors on CPU
            if (cpu_tensor2.defined() && cpu_tensor3.defined()) {
                bool same_device = cpu_tensor2.device() == cpu_tensor3.device();
                bool same_dtype = cpu_tensor2.dtype() == cpu_tensor3.dtype();
                bool same_shape = cpu_tensor2.sizes() == cpu_tensor3.sizes();
                
                // Both should be on CPU
                bool both_cpu = cpu_tensor2.device().is_cpu() && cpu_tensor3.device().is_cpu();
            }
        }
        
        // Test cpu() on different tensor types
        if (Size >= 4) {
            // Test with a contiguous tensor
            torch::Tensor contig = tensor.contiguous().cpu();
            
            // Test with a view (non-contiguous potentially)
            if (tensor.numel() > 1) {
                try {
                    torch::Tensor view = tensor.view({-1}).cpu();
                } catch (...) {
                    // Shape might not be compatible, ignore
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}