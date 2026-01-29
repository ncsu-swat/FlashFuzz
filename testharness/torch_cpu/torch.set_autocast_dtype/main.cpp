#include "fuzzer_utils.h"
#include <iostream>
#include <ATen/autocast_mode.h>

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
        
        // Need at least 2 bytes for device type and dtype selection
        if (Size < 2) {
            return 0;
        }
        
        // Parse the device type (CPU or CUDA)
        uint8_t device_selector = Data[offset++];
        at::DeviceType device_type;
        
        // Select between CPU and CUDA device types for autocast
        switch (device_selector % 3) {
            case 0:
                device_type = at::DeviceType::CPU;
                break;
            case 1:
                device_type = at::DeviceType::CUDA;
                break;
            default:
                device_type = at::DeviceType::CPU;
                break;
        }
        
        // Parse the dtype to use for autocast
        // Autocast typically uses reduced precision types
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType autocast_dtype;
        
        switch (dtype_selector % 4) {
            case 0:
                autocast_dtype = torch::kFloat16;
                break;
            case 1:
                autocast_dtype = torch::kBFloat16;
                break;
            case 2:
                autocast_dtype = torch::kFloat32;
                break;
            default:
                autocast_dtype = torch::kFloat16;
                break;
        }
        
        // Save the original autocast dtype to restore later
        torch::ScalarType original_dtype;
        try {
            original_dtype = at::autocast::get_autocast_dtype(device_type);
        } catch (...) {
            // If getting original fails, use a default
            original_dtype = torch::kFloat16;
        }
        
        // Call the actual API being tested: set_autocast_dtype
        at::autocast::set_autocast_dtype(device_type, autocast_dtype);
        
        // Verify the dtype was set correctly by getting it back
        torch::ScalarType retrieved_dtype = at::autocast::get_autocast_dtype(device_type);
        
        // The retrieved dtype should match what we set
        if (retrieved_dtype != autocast_dtype) {
            // This would indicate unexpected behavior
            std::cerr << "Dtype mismatch after set_autocast_dtype" << std::endl;
        }
        
        // Test with different dtypes in sequence if we have more data
        if (offset < Size) {
            uint8_t second_dtype_selector = Data[offset++];
            torch::ScalarType second_dtype;
            
            switch (second_dtype_selector % 4) {
                case 0:
                    second_dtype = torch::kFloat16;
                    break;
                case 1:
                    second_dtype = torch::kBFloat16;
                    break;
                case 2:
                    second_dtype = torch::kFloat32;
                    break;
                default:
                    second_dtype = torch::kFloat16;
                    break;
            }
            
            // Set a different dtype
            at::autocast::set_autocast_dtype(device_type, second_dtype);
            
            // Verify again
            torch::ScalarType second_retrieved = at::autocast::get_autocast_dtype(device_type);
            if (second_retrieved != second_dtype) {
                std::cerr << "Second dtype mismatch after set_autocast_dtype" << std::endl;
            }
        }
        
        // Test setting autocast dtype for multiple device types
        if (offset < Size) {
            at::DeviceType other_device = (device_type == at::DeviceType::CPU) 
                ? at::DeviceType::CUDA 
                : at::DeviceType::CPU;
            
            uint8_t other_dtype_selector = Data[offset++];
            torch::ScalarType other_dtype = (other_dtype_selector % 2 == 0) 
                ? torch::kFloat16 
                : torch::kBFloat16;
            
            try {
                at::autocast::set_autocast_dtype(other_device, other_dtype);
                at::autocast::get_autocast_dtype(other_device);
            } catch (...) {
                // Silently ignore if setting for other device fails
            }
        }
        
        // Restore original dtype
        try {
            at::autocast::set_autocast_dtype(device_type, original_dtype);
        } catch (...) {
            // Ignore restoration errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}