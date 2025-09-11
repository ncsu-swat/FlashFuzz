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
        
        // Need at least 1 byte for device type and 1 byte for index
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to test with device operations
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Default tensor if we've consumed all data
            tensor = torch::ones({2, 3});
        }
        
        // Parse device type (cpu, cuda, etc.)
        uint8_t device_type_byte = (offset < Size) ? Data[offset++] : 0;
        uint8_t device_index_byte = (offset < Size) ? Data[offset++] : 0;
        
        // Map device_type_byte to a device type
        std::string device_type;
        switch (device_type_byte % 4) {
            case 0:
                device_type = "cpu";
                break;
            case 1:
                device_type = "cuda";
                break;
            case 2:
                device_type = "mkldnn";
                break;
            case 3:
                device_type = "opengl";
                break;
        }
        
        // Map device_index_byte to a device index
        int device_index = device_index_byte % 8; // Limit to 0-7 range
        
        // Test different ways to create and use torch::Device
        
        // 1. Create device from string
        try {
            torch::Device device1(device_type);
            auto tensor_on_device1 = tensor.to(device1);
        } catch (...) {
            // Device might not be available, continue
        }
        
        // 2. Create device from string with index
        try {
            std::string device_str = device_type + ":" + std::to_string(device_index);
            torch::Device device2(device_str);
            auto tensor_on_device2 = tensor.to(device2);
        } catch (...) {
            // Device might not be available, continue
        }
        
        // 3. Create device from type and index
        try {
            std::string device_str = device_type + ":" + std::to_string(device_index);
            torch::Device device3(device_str);
            auto tensor_on_device3 = tensor.to(device3);
        } catch (...) {
            // Device might not be available, continue
        }
        
        // 4. Test device properties
        try {
            std::string device_str = device_type + ":" + std::to_string(device_index);
            torch::Device device4(device_str);
            std::string device_str_out = device4.str();
            torch::DeviceType device_type_enum = device4.type();
            int index = device4.index();
            bool is_cuda = device4.is_cuda();
            bool is_cpu = device4.is_cpu();
        } catch (...) {
            // Device might not be available, continue
        }
        
        // 5. Test device equality
        try {
            std::string device_str_a = device_type + ":" + std::to_string(device_index);
            std::string device_str_b = device_type + ":" + std::to_string(device_index);
            std::string device_str_c = device_type + ":" + std::to_string((device_index + 1) % 8);
            
            torch::Device device5a(device_str_a);
            torch::Device device5b(device_str_b);
            torch::Device device5c(device_str_c);
            
            bool equal = (device5a == device5b);
            bool not_equal = (device5a != device5c);
        } catch (...) {
            // Device might not be available, continue
        }
        
        // 6. Test moving tensors between devices
        try {
            // Create two different devices
            torch::Device cpu_device("cpu");
            std::string other_device_str = device_type + ":" + std::to_string(device_index);
            torch::Device other_device(other_device_str);
            
            // Move tensor to first device
            auto tensor_on_cpu = tensor.to(cpu_device);
            
            // Move to second device if different from CPU
            if (device_type != "cpu") {
                auto tensor_on_other = tensor_on_cpu.to(other_device);
                // Move back to CPU
                auto back_to_cpu = tensor_on_other.to(cpu_device);
            }
        } catch (...) {
            // Device might not be available, continue
        }
        
        // 7. Test device with options
        try {
            std::string device_str = device_type + ":" + std::to_string(device_index);
            torch::Device device7(device_str);
            torch::TensorOptions options = torch::TensorOptions().device(device7);
            auto new_tensor = torch::ones({3, 4}, options);
        } catch (...) {
            // Device might not be available, continue
        }
        
        // 8. Test device with invalid inputs
        try {
            // Create device with invalid type
            if (offset < Size) {
                std::string invalid_type = "invalid_device_type";
                torch::Device invalid_device(invalid_type);
            }
        } catch (...) {
            // Expected to throw
        }
        
        // 9. Test device with negative index
        try {
            if (offset < Size) {
                int negative_index = -1;
                std::string negative_device_str = device_type + ":" + std::to_string(negative_index);
                torch::Device negative_device(negative_device_str);
            }
        } catch (...) {
            // May throw depending on implementation
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
