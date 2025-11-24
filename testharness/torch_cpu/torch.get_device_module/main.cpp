#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// Fuzz target: torch.get_device_module

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
                    int available = torch::cuda::device_count();
                    if (available <= 0) {
                        available = 1;
                    }
                    if (offset < Size) {
                        device_index = static_cast<int>(Data[offset++]) % available;
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
        
        // Test device functionality based on type
        if (device.is_cuda() && torch::cuda::is_available()) {
            int device_count = torch::cuda::device_count();
            if (device_count <= 0) {
                device_count = 1;
            }
            auto index = device.index();
            if (index < 0 || index >= device_count) {
                index = 0;
            }
            // Synchronize with the chosen device index
            torch::cuda::synchronize(index);
        }
        else if (device.type() == torch::kMPS && torch::mps::is_available()) {
            torch::mps::synchronize();
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
