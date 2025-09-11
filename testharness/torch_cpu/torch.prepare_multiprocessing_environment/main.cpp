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
        
        // Create a tensor to test operations
        if (Size > 0) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform a simple operation to verify tensor functionality
            torch::Tensor result = tensor + 1;
            
            // Test if we can move tensors between CPU and GPU if available
            if (torch::cuda::is_available()) {
                torch::Tensor gpu_tensor = tensor.to(torch::kCUDA);
                torch::Tensor cpu_tensor = gpu_tensor.to(torch::kCPU);
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
