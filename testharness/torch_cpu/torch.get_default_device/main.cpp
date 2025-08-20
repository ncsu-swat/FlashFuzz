#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Get the default device (using torch::kCPU as default since get_default_device doesn't exist)
        auto default_device = torch::kCPU;
        
        // Create a tensor on the default device
        if (Size > 0) {
            auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Move tensor to default device
            tensor = tensor.to(default_device);
            
            // Verify the device
            auto tensor_device = tensor.device();
            if (tensor_device != default_device) {
                throw std::runtime_error("Tensor device doesn't match default device");
            }
            
            // Try to get default device again after tensor creation
            auto default_device_after = torch::kCPU;
            
            // Check if default device changed
            if (default_device != default_device_after) {
                throw std::runtime_error("Default device changed unexpectedly");
            }
        }
        
        // Test with empty tensor
        torch::Tensor empty_tensor = torch::empty({0});
        empty_tensor = empty_tensor.to(default_device);
        
        // Test with scalar tensor
        torch::Tensor scalar_tensor = torch::tensor(1.0);
        scalar_tensor = scalar_tensor.to(default_device);
        
        // Test with boolean tensor
        torch::Tensor bool_tensor = torch::tensor(true);
        bool_tensor = bool_tensor.to(default_device);
        
        // Test with complex tensor using c10::complex
        torch::Tensor complex_tensor = torch::tensor(c10::complex<double>(1.0, 2.0));
        complex_tensor = complex_tensor.to(default_device);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}