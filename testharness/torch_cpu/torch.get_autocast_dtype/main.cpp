#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Parse device type from input data
        torch::DeviceType device_type = torch::kCPU;
        if (offset < Size) {
            uint8_t device_byte = Data[offset++];
            device_type = (device_byte % 2 == 0) ? torch::kCPU : torch::kCUDA;
        }
        
        // Parse autocast enabled flag
        bool enabled = false;
        if (offset < Size) {
            uint8_t enabled_byte = Data[offset++];
            enabled = (enabled_byte % 2 == 0);
        }
        
        // Parse device type for get_autocast_dtype
        torch::DeviceType target_device = torch::kCPU;
        if (offset < Size) {
            uint8_t target_device_byte = Data[offset++];
            target_device = (target_device_byte % 2 == 0) ? torch::kCPU : torch::kCUDA;
        }
        
        // Create a tensor to test with
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a simple tensor
            tensor = torch::ones({2, 2});
        }
        
        // Set autocast state
        torch::set_autocast_enabled(enabled);
        
        // Set autocast dtype for CPU
        if (offset < Size) {
            uint8_t cpu_dtype_byte = Data[offset++];
            torch::ScalarType cpu_dtype = fuzzer_utils::parseDataType(cpu_dtype_byte);
            torch::set_autocast_cpu_dtype(cpu_dtype);
        }
        
        // Set autocast dtype for CUDA
        if (offset < Size) {
            uint8_t cuda_dtype_byte = Data[offset++];
            torch::ScalarType cuda_dtype = fuzzer_utils::parseDataType(cuda_dtype_byte);
            torch::set_autocast_gpu_dtype(cuda_dtype);
        }
        
        // Call get_autocast_dtype
        torch::ScalarType result_dtype = torch::get_autocast_dtype(target_device);
        
        // Test with the tensor to ensure the autocast settings are applied
        torch::Tensor output;
        if (tensor.defined()) {
            // Move tensor to the target device if CUDA is available and requested
            if (target_device == torch::kCUDA && torch::cuda::is_available()) {
                tensor = tensor.to(torch::kCUDA);
            }
            
            // Perform a simple operation with autocast
            torch::AutocastGuard guard(target_device);
            output = tensor + tensor;
        }
        
        // Reset autocast state to avoid affecting other tests
        torch::set_autocast_enabled(false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}