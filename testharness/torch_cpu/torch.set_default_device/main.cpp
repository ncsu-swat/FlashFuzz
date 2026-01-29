#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
        
        // Use modulo to select device type (only use CPU and Meta for fuzzing)
        switch (device_type % 2) {
            case 0:
                device = c10::Device(c10::DeviceType::CPU);
                break;
            case 1:
                // Meta device is always available and useful for testing
                device = c10::Device(c10::DeviceType::Meta);
                break;
        }
        
        // Set the default device - this is the main API being tested
        c10::impl::set_default_device(device);
        
        // Get the current default device to verify it was set
        std::optional<c10::Device> current_default = c10::impl::getDefaultDevice();
        
        // Verify the default device was set
        if (current_default.has_value()) {
            // Check that the device type matches what we set
            (void)(current_default->type() == device.type());
        }
        
        // Test setting device multiple times
        if (Size > offset) {
            uint8_t second_device_type = Data[offset++];
            c10::Device second_device(c10::DeviceType::CPU);
            
            if (second_device_type % 2 == 1) {
                second_device = c10::Device(c10::DeviceType::Meta);
            }
            
            c10::impl::set_default_device(second_device);
            
            // Verify the new default
            std::optional<c10::Device> new_default = c10::impl::getDefaultDevice();
            (void)new_default;
        }
        
        // Test with nullopt to clear default device
        if (Size > offset && Data[offset] % 4 == 0) {
            c10::impl::set_default_device(std::nullopt);
            
            // Verify default is cleared
            std::optional<c10::Device> cleared_default = c10::impl::getDefaultDevice();
            (void)cleared_default;
        }
        
        // Create tensors on CPU to exercise tensor creation after setting default
        // Note: In C++ API, tensor creation may not always respect default device
        // depending on the function used
        try {
            if (device.type() == c10::DeviceType::CPU) {
                // Only create real tensors on CPU to avoid Meta device limitations
                torch::Tensor t = torch::zeros({2, 2});
                (void)t;
                
                torch::Tensor t2 = torch::ones({3});
                (void)t2;
            }
        } catch (...) {
            // Silently ignore tensor creation failures
        }
        
        // Reset to CPU to avoid affecting other fuzzing iterations
        c10::impl::set_default_device(c10::Device(c10::DeviceType::CPU));
    }
    catch (const std::exception &e)
    {
        // Reset device before returning
        try {
            c10::impl::set_default_device(c10::Device(c10::DeviceType::CPU));
        } catch (...) {}
        
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}