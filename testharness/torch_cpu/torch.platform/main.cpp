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
        
        // Get platform information using preprocessor macros
        std::string platform;
        #ifdef _WIN32
            platform = "windows";
        #elif __APPLE__
            platform = "macos";
        #elif __linux__
            platform = "linux";
        #else
            platform = "unknown";
        #endif
        
        // Test with different tensor types and shapes
        if (Size > 2) {
            // Create a tensor to test with platform info
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test platform-specific operations
            if (platform == "linux") {
                // Test Linux-specific operations
                auto result = tensor.clone();
            } else if (platform == "windows") {
                // Test Windows-specific operations
                auto result = tensor.clone();
            } else if (platform == "macos") {
                // Test macOS-specific operations
                auto result = tensor.clone();
            }
            
            // Test platform with CUDA if available
            if (torch::cuda::is_available()) {
                auto cuda_tensor = tensor.cuda();
                auto cpu_tensor = cuda_tensor.cpu();
            }
        }
        
        // Test platform with different device types
        torch::Device cpu_device(torch::kCPU);
        std::string device_str = cpu_device.str();
        
        if (torch::cuda::is_available() && Size > 0) {
            int device_idx = Data[0] % torch::cuda::device_count();
            torch::Device cuda_device(torch::kCUDA, device_idx);
            device_str = cuda_device.str();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
