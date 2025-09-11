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
        
        // Check if autocast cache is enabled
        bool is_enabled = torch::is_autocast_cache_enabled();
        
        // Try with different device types
        bool is_enabled_cpu = torch::is_autocast_cache_enabled(torch::kCPU);
        
        // If CUDA is available, check for CUDA device
        if (torch::cuda::is_available()) {
            bool is_enabled_cuda = torch::is_autocast_cache_enabled(torch::kCUDA);
        }
        
        // Create a tensor to test with autocast operations
        if (Size > 0) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Enable autocast cache for CPU
            torch::set_autocast_cache_enabled(true);
            
            // Check if autocast cache is enabled after enabling
            bool is_enabled_after = torch::is_autocast_cache_enabled();
            
            // Perform some operation with the tensor
            torch::Tensor result = tensor * 2.0;
            
            // Disable autocast cache
            torch::set_autocast_cache_enabled(false);
            
            // Check if autocast cache is disabled after disabling
            bool is_enabled_final = torch::is_autocast_cache_enabled();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
