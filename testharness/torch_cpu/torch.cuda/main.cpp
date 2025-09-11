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
        
        // Skip if data is too small
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test CUDA availability
        bool is_cuda_available = torch::cuda::is_available();
        
        // Only proceed with CUDA operations if CUDA is available
        if (is_cuda_available) {
            // Get device count
            int device_count = torch::cuda::device_count();
            
            // Test device properties if devices are available
            if (device_count > 0) {
                // Test device synchronization
                torch::cuda::synchronize();
                
                // Test moving tensor to CUDA
                torch::Tensor cuda_tensor = tensor.cuda();
                
                // Test operations on CUDA tensor
                torch::Tensor result = cuda_tensor + 1;
                
                // Test moving back to CPU
                torch::Tensor cpu_tensor = result.cpu();
            }
        }
        
        // Test is_available with different device types
        bool is_cudnn_available = torch::cuda::cudnn_is_available();
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
