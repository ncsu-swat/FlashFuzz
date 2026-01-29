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
        // Need at least 2 bytes for device type and index
        if (Size < 2) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse device type and index from the first two bytes
        uint8_t device_type_byte = Data[offset++];
        uint8_t device_index_byte = Data[offset++];
        
        // Create a tensor to test with device operations
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Default tensor if we've consumed all data
            tensor = torch::ones({2, 3});
        }
        
        // Map device_type_byte to a valid device type
        // Only use device types that exist in PyTorch C++ frontend
        std::string device_type;
        switch (device_type_byte % 3) {
            case 0:
                device_type = "cpu";
                break;
            case 1:
                device_type = "cuda";
                break;
            case 2:
                device_type = "meta";
                break;
        }
        
        // Map device_index_byte to a device index
        int device_index = device_index_byte % 8; // Limit to 0-7 range
        
        // Test different ways to create and use torch::Device
        
        // 1. Create device from string (no index)
        try {
            torch::Device device1(device_type);
            // Test device properties
            (void)device1.str();
            (void)device1.type();
            (void)device1.has_index();
            (void)device1.is_cuda();
            (void)device1.is_cpu();
            (void)device1.is_meta();
            
            // Only move to CPU or meta devices since CUDA may not be available
            if (device1.is_cpu() || device1.is_meta()) {
                auto tensor_on_device1 = tensor.to(device1);
            }
        } catch (...) {
            // Device might not be available, continue
        }
        
        // 2. Create device from string with index
        try {
            std::string device_str = device_type + ":" + std::to_string(device_index);
            torch::Device device2(device_str);
            (void)device2.index();
            
            if (device2.is_cpu()) {
                auto tensor_on_device2 = tensor.to(device2);
            }
        } catch (...) {
            // Device might not be available, continue
        }
        
        // 3. Create device from DeviceType enum directly
        try {
            torch::Device device3(torch::kCPU);
            auto tensor_on_cpu = tensor.to(device3);
            (void)tensor_on_cpu.device();
        } catch (...) {
            // Continue on failure
        }
        
        // 4. Create device from DeviceType enum with index
        try {
            torch::Device device4(torch::kCPU, 0);
            (void)device4.str();
            (void)device4.index();
        } catch (...) {
            // Continue on failure
        }
        
        // 5. Test device equality and comparison
        try {
            torch::Device cpu_device1(torch::kCPU);
            torch::Device cpu_device2("cpu");
            torch::Device cpu_device3(torch::kCPU, 0);
            
            bool equal1 = (cpu_device1 == cpu_device2);
            bool equal2 = (cpu_device1 == cpu_device3);
            (void)equal1;
            (void)equal2;
            
            // Test inequality
            torch::Device meta_device(torch::kMeta);
            bool not_equal = (cpu_device1 != meta_device);
            (void)not_equal;
        } catch (...) {
            // Continue on failure
        }
        
        // 6. Test device with TensorOptions
        try {
            torch::Device cpu_device(torch::kCPU);
            torch::TensorOptions options = torch::TensorOptions().device(cpu_device);
            auto new_tensor = torch::ones({3, 4}, options);
            
            // Verify device
            torch::Device result_device = new_tensor.device();
            (void)result_device.is_cpu();
        } catch (...) {
            // Continue on failure
        }
        
        // 7. Test tensor.device() accessor
        try {
            torch::Device tensor_device = tensor.device();
            (void)tensor_device.str();
            (void)tensor_device.type();
            (void)tensor_device.is_cpu();
        } catch (...) {
            // Continue on failure
        }
        
        // 8. Test device set_index (if available)
        try {
            torch::Device device8(torch::kCPU);
            // Check if device supports index operations
            if (device8.has_index()) {
                (void)device8.index();
            }
        } catch (...) {
            // Continue on failure
        }
        
        // 9. Test creating device from another device
        try {
            torch::Device original(torch::kCPU);
            torch::Device copy = original;
            bool same = (original == copy);
            (void)same;
        } catch (...) {
            // Continue on failure
        }
        
        // 10. Test device hash (for use in containers)
        try {
            torch::Device device10(torch::kCPU);
            std::hash<torch::Device> hasher;
            size_t hash_val = hasher(device10);
            (void)hash_val;
        } catch (...) {
            // Hash might not be available in all versions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}