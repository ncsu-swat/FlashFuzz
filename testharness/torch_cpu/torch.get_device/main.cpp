#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor with device specification
        torch::Tensor tensor;
        
        offset++; // Skip first byte (was used for CUDA selection, keep for consistency)
        
        // Create the tensor on CPU
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test get_device functionality
            // For CPU tensors, get_device() returns -1
            int device = tensor.get_device();
            (void)device; // Suppress unused variable warning
            
            // Test additional device-related operations
            bool is_cuda = tensor.is_cuda();
            bool is_cpu = tensor.is_cpu();
            (void)is_cuda;
            (void)is_cpu;
            
            // Try to get device for a view of the tensor
            try {
                if (tensor.numel() > 0) {
                    torch::Tensor view_tensor = tensor.view({-1});
                    int view_device = view_tensor.get_device();
                    (void)view_device;
                }
            } catch (...) {
                // View may fail for certain tensor configurations, ignore
            }
            
            // Try to get device for a slice of the tensor if possible
            try {
                if (tensor.dim() > 0 && tensor.size(0) > 1) {
                    torch::Tensor slice_tensor = tensor.slice(0, 0, 1);
                    int slice_device = slice_tensor.get_device();
                    (void)slice_device;
                }
            } catch (...) {
                // Slice may fail, ignore
            }
            
            // Try to get device for a transposed tensor if possible
            try {
                if (tensor.dim() >= 2) {
                    torch::Tensor transposed = tensor.transpose(0, 1);
                    int transposed_device = transposed.get_device();
                    (void)transposed_device;
                }
            } catch (...) {
                // Transpose may fail, ignore
            }
            
            // Try to get device for a tensor after in-place operations
            try {
                if (tensor.numel() > 0 && tensor.is_floating_point()) {
                    torch::Tensor clone_tensor = tensor.clone();
                    clone_tensor.mul_(2.0);
                    int clone_device = clone_tensor.get_device();
                    (void)clone_device;
                }
            } catch (...) {
                // In-place op may fail, ignore
            }
            
            // Additional coverage: test device() method which returns Device object
            torch::Device dev = tensor.device();
            bool is_cpu_device = dev.is_cpu();
            (void)is_cpu_device;
            
            // Test contiguous tensor device
            torch::Tensor contig = tensor.contiguous();
            int contig_device = contig.get_device();
            (void)contig_device;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}