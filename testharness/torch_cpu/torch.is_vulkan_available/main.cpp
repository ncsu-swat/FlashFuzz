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
        
        // Check if Vulkan is available
        bool is_vulkan_available = torch::is_vulkan_available();
        
        // Create a tensor to test with Vulkan if available
        if (Size > 0 && is_vulkan_available) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // If Vulkan is available, try to move tensor to Vulkan device
            if (tensor.defined()) {
                try {
                    torch::Tensor vulkan_tensor = tensor.vulkan();
                    
                    // Perform some operations on the Vulkan tensor
                    torch::Tensor result = vulkan_tensor + 1.0;
                    
                    // Move back to CPU for verification
                    torch::Tensor cpu_result = result.cpu();
                } catch (const std::exception& e) {
                    // Catch exceptions from Vulkan operations but don't discard the input
                }
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
