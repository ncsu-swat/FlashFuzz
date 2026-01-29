#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        // Check if Vulkan is available - this is the main API under test
        bool is_vulkan_available = torch::is_vulkan_available();
        
        // Use the fuzzer data to create tensors and exercise Vulkan operations
        // if Vulkan is available
        if (Size > 0 && is_vulkan_available) {
            size_t offset = 0;
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (tensor.defined()) {
                try {
                    // Move tensor to Vulkan device using the proper API
                    torch::Tensor vulkan_tensor = tensor.to(torch::kVulkan);
                    
                    // Perform some operations on the Vulkan tensor
                    torch::Tensor result = vulkan_tensor + 1.0;
                    
                    // Move back to CPU for verification
                    torch::Tensor cpu_result = result.to(torch::kCPU);
                } catch (const std::exception& e) {
                    // Silently catch exceptions from Vulkan operations
                    // (e.g., unsupported dtype, Vulkan backend limitations)
                }
            }
        }
        
        // Even if Vulkan is not available, we still exercised the API
        // by calling is_vulkan_available()
        
        // Additionally, exercise the API with different call patterns
        // to ensure the function is robust
        volatile bool check1 = torch::is_vulkan_available();
        volatile bool check2 = torch::is_vulkan_available();
        
        // Consistency check - should always return the same value
        if (check1 != check2) {
            std::cerr << "Inconsistent is_vulkan_available() results" << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}