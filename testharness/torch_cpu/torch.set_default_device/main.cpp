#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte to determine device type
        if (Size < 1) {
            return 0;
        }
        
        // Parse device type from input data
        uint8_t device_type = Data[offset++];
        
        // Create a device based on the input
        c10::Device device(c10::DeviceType::CPU);
        
        // Use modulo to select device type
        switch (device_type % 3) {
            case 0:
                device = c10::Device(c10::DeviceType::CPU);
                break;
            case 1:
                if (torch::cuda::is_available()) {
                    device = c10::Device(c10::DeviceType::CUDA, 0);
                } else {
                    device = c10::Device(c10::DeviceType::CPU);
                }
                break;
            case 2:
                device = c10::Device(c10::DeviceType::Meta);
                break;
        }
        
        // Set the default device
        c10::impl::set_default_device(device);
        
        // Create a tensor to verify the default device is set correctly
        if (Size > offset) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Create another tensor without specifying device to check if it uses the default
            torch::Tensor new_tensor = torch::ones_like(tensor);
            
            // Verify the new tensor is on the expected device
            if (new_tensor.device().type() != device.type()) {
                throw std::runtime_error("Default device not applied correctly");
            }
            
            // Try creating a tensor with explicit options
            torch::TensorOptions options;
            torch::Tensor explicit_tensor = torch::ones(tensor.sizes(), options);
            
            // Verify this tensor is also on the default device
            if (explicit_tensor.device().type() != device.type()) {
                throw std::runtime_error("Default device not applied to explicit options");
            }
            
            // Test with empty tensor
            torch::Tensor empty_tensor = torch::empty({0});
            if (empty_tensor.device().type() != device.type()) {
                throw std::runtime_error("Default device not applied to empty tensor");
            }
            
            // Test with scalar tensor
            torch::Tensor scalar = torch::tensor(3.14);
            if (scalar.device().type() != device.type()) {
                throw std::runtime_error("Default device not applied to scalar tensor");
            }
        }
        
        // Reset to CPU to avoid affecting other tests
        c10::impl::set_default_device(c10::Device(c10::DeviceType::CPU));
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}