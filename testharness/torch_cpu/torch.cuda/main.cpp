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
        // Skip if data is too small
        if (Size < 1) {
            return 0;
        }
        
        // Use input data to vary which APIs we call
        uint8_t selector = Data[0];
        
        // Test CUDA availability query APIs
        // These work even without CUDA hardware - they just return false/0
        
        if (selector & 0x01) {
            // Test CUDA availability check
            bool is_cuda_available = torch::cuda::is_available();
            (void)is_cuda_available; // Prevent unused warning
        }
        
        if (selector & 0x02) {
            // Test device count query
            int64_t device_count = torch::cuda::device_count();
            (void)device_count;
        }
        
        if (selector & 0x04) {
            // Test cuDNN availability check
            bool is_cudnn_available = torch::cuda::cudnn_is_available();
            (void)is_cudnn_available;
        }
        
        if (selector & 0x08) {
            // Test if CUDA is built (compiled with CUDA support)
            bool is_built = torch::cuda::is_available(); // Will be false if not built with CUDA
            (void)is_built;
        }
        
        // If CUDA is actually available, test more operations
        if (torch::cuda::is_available()) {
            int64_t device_count = torch::cuda::device_count();
            
            if (device_count > 0 && Size >= 4) {
                size_t offset = 1;
                
                // Create a tensor from remaining input data
                torch::Tensor tensor = fuzzer_utils::createTensor(Data + 1, Size - 1, offset);
                
                try {
                    // Test synchronization
                    torch::cuda::synchronize();
                    
                    // Test moving tensor to CUDA and back
                    torch::Tensor cuda_tensor = tensor.to(torch::kCUDA);
                    torch::Tensor result = cuda_tensor + 1.0;
                    torch::Tensor cpu_tensor = result.to(torch::kCPU);
                    
                    // Synchronize after operations
                    torch::cuda::synchronize();
                }
                catch (const c10::Error &e) {
                    // CUDA operations may fail for various reasons (OOM, etc.)
                    // This is expected behavior, not a bug
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