#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/autocast_mode.h>
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

namespace
{
    // Limit autocast dtype choices to valid lower-precision float options.
    torch::ScalarType choose_autocast_dtype(uint8_t selector)
    {
        switch (selector % 3)
        {
        case 0:
            return torch::kFloat16;
        case 1:
            return torch::kBFloat16;
        default:
            return torch::kFloat;
        }
    }
} // namespace

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Parse device type from input data
        at::DeviceType device_type = at::kCPU;
        if (offset < Size) {
            uint8_t device_byte = Data[offset++];
            device_type = (device_byte % 2 == 0) ? at::kCPU : at::kCUDA;
        }
        
        // Parse autocast enabled flag
        bool enabled = false;
        if (offset < Size) {
            uint8_t enabled_byte = Data[offset++];
            enabled = (enabled_byte % 2 == 0);
        }
        
        // Parse device type for get_autocast_dtype
        at::DeviceType target_device = at::kCPU;
        if (offset < Size) {
            uint8_t target_device_byte = Data[offset++];
            target_device = (target_device_byte % 2 == 0) ? at::kCPU : at::kCUDA;
        }
        
        // Create a tensor to test with
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a simple tensor
            tensor = torch::ones({2, 2});
        }
        if (tensor.defined() && device_type == at::kCUDA && torch::cuda::is_available()) {
            tensor = tensor.to(torch::kCUDA);
        }
        
        // Set autocast state for the chosen device
        at::autocast::set_autocast_enabled(target_device, enabled);
        
        // Set autocast dtype for CPU
        if (offset < Size) {
            uint8_t cpu_dtype_byte = Data[offset++];
            torch::ScalarType cpu_dtype = choose_autocast_dtype(cpu_dtype_byte);
            at::autocast::set_autocast_dtype(at::kCPU, cpu_dtype);
        }
        
        // Set autocast dtype for CUDA
        if (offset < Size) {
            uint8_t cuda_dtype_byte = Data[offset++];
            torch::ScalarType cuda_dtype = choose_autocast_dtype(cuda_dtype_byte);
            at::autocast::set_autocast_dtype(at::kCUDA, cuda_dtype);
        }
        
        // Call get_autocast_dtype (target API: torch.get_autocast_dtype)
        torch::ScalarType result_dtype = at::autocast::get_autocast_dtype(target_device);
        
        // Test with the tensor to ensure the autocast settings are applied
        torch::Tensor output;
        if (tensor.defined()) {
            // Move tensor to the target device if CUDA is available and requested
            if (target_device == at::kCUDA && torch::cuda::is_available()) {
                tensor = tensor.to(torch::kCUDA);
            }
            
            // Perform a simple operation; convert to the chosen autocast dtype to exercise it
            torch::Tensor cast_tensor = tensor.to(result_dtype);
            output = cast_tensor + cast_tensor;
            (void)output.sum();
        }
        
        // Reset autocast state to avoid affecting other tests
        at::autocast::set_autocast_enabled(target_device, false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
