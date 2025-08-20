#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor with device specification
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try different device options if there's more data
        if (offset + 1 < Size) {
            uint8_t device_type = Data[offset++];
            
            // Use the device type to determine which device to use
            // We'll use modulo to ensure we get a valid device type
            device_type = device_type % 3; // 0: CPU, 1: CUDA (if available), 2: MPS (if available)
            
            torch::Device device = torch::Device(torch::kCPU);
            
            if (device_type == 0) {
                device = torch::Device(torch::kCPU);
            } 
            else if (device_type == 1) {
                if (torch::cuda::is_available()) {
                    // If we have more data, use it to select a CUDA device index
                    int device_index = 0;
                    if (offset < Size) {
                        device_index = static_cast<int>(Data[offset++]) % torch::cuda::device_count();
                    }
                    device = torch::Device(torch::kCUDA, device_index);
                } else {
                    device = torch::Device(torch::kCPU);
                }
            }
            else if (device_type == 2) {
                if (torch::mps::is_available()) {
                    device = torch::Device(torch::kMPS);
                } else {
                    device = torch::Device(torch::kCPU);
                }
            }
            
            // Move tensor to the selected device
            tensor = tensor.to(device);
        }
        
        // Test tensor device properties
        auto device = tensor.device();
        auto device_type = device.type();
        auto device_index = device.index();
        
        // Test device functionality based on type
        if (device.is_cuda() && torch::cuda::is_available()) {
            int device_count = torch::cuda::device_count();
            int current_device = torch::cuda::current_device();
            
            // Test setting current device if possible
            if (device_count > 1) {
                int new_device = (current_device + 1) % device_count;
                torch::cuda::set_device(new_device);
            }
            
            // Test synchronization
            torch::cuda::synchronize();
            
            // Test device properties
            auto properties = torch::cuda::getDeviceProperties(device.index());
        }
        
        // Test basic tensor operations on the device
        auto result = tensor + 1.0;
        auto sum = torch::sum(result);
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}