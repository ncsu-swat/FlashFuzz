#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/csrc/autograd/autocast_mode.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Check if we have enough data to proceed
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to use in the context
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            tensor = torch::ones({1, 1});
        }
        
        // Extract a boolean from the data to determine if we should use enabled=true/false
        bool enabled = false;
        if (offset < Size) {
            enabled = static_cast<bool>(Data[offset++] & 0x1);
        }
        
        // Extract a device type from the data
        c10::DeviceType device_type = c10::kCPU;
        if (offset < Size) {
            uint8_t device_selector = Data[offset++];
            // Only use kCUDA if CUDA is available
            if ((device_selector & 0x1) && torch::cuda::is_available()) {
                device_type = c10::kCUDA;
            }
        }
        
        // Call autocast_increment_nesting
        torch::autocast::increment_nesting(device_type);
        
        // Perform some operations inside the autocast context
        torch::Tensor result = tensor + 1.0;
        
        // Decrement nesting to balance the increment
        torch::autocast::decrement_nesting(device_type);
        
        // Try with different device types
        if (offset < Size && torch::cuda::is_available()) {
            torch::autocast::increment_nesting(c10::kCUDA);
            torch::Tensor cuda_result = tensor.to(torch::kCUDA) + 1.0;
            torch::autocast::decrement_nesting(c10::kCUDA);
        }
        
        // Try with different enabled values
        torch::autocast::increment_nesting(device_type);
        torch::Tensor another_result = tensor * 2.0;
        torch::autocast::decrement_nesting(device_type);
        
        // Try nested calls
        torch::autocast::increment_nesting(device_type);
        torch::autocast::increment_nesting(device_type);
        torch::Tensor nested_result = tensor.pow(2);
        torch::autocast::decrement_nesting(device_type);
        torch::autocast::decrement_nesting(device_type);
        
        // Try with different tensor types
        if (offset + 1 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            torch::autocast::increment_nesting(device_type);
            torch::Tensor mixed_result = tensor + another_tensor;
            torch::autocast::decrement_nesting(device_type);
        }
        
        // Try with extreme nesting levels
        uint8_t nesting_level = 1;
        if (offset < Size) {
            nesting_level = Data[offset++] % 10; // Limit to reasonable number
        }
        
        for (uint8_t i = 0; i < nesting_level; i++) {
            torch::autocast::increment_nesting(device_type);
        }
        
        torch::Tensor deep_nested_result = tensor.sin();
        
        for (uint8_t i = 0; i < nesting_level; i++) {
            torch::autocast::decrement_nesting(device_type);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}