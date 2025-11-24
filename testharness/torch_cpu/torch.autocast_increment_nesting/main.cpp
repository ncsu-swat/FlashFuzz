#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/autocast_mode.h>
#include <iostream> // For cerr
#include <tuple>    // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
        at::DeviceType device_type = at::kCPU;
        if (offset < Size) {
            uint8_t device_selector = Data[offset++];
            // Only use kCUDA if CUDA is available
            if ((device_selector & 0x1) && torch::cuda::is_available()) {
                device_type = at::kCUDA;
            }
        }
        
        at::autocast::set_autocast_enabled(device_type, enabled);
        
        // Call autocast_increment_nesting
        at::autocast::increment_nesting();
        
        // Perform some operations inside the autocast context
        torch::Tensor result = tensor + 1.0;
        (void)result.sum(); // ensure the result is used
        
        // Decrement nesting to balance the increment
        at::autocast::decrement_nesting();
        
        // Try with different device types
        if (offset < Size && torch::cuda::is_available()) {
            at::autocast::set_autocast_enabled(at::kCUDA, enabled);
            at::autocast::increment_nesting();
            torch::Tensor cuda_result = tensor.to(torch::kCUDA) + 1.0;
            (void)cuda_result.sum();
            at::autocast::decrement_nesting();
        }
        
        // Try with different enabled values
        at::autocast::increment_nesting();
        torch::Tensor another_result = tensor * 2.0;
        (void)another_result.sum();
        at::autocast::decrement_nesting();
        
        // Try nested calls
        at::autocast::increment_nesting();
        at::autocast::increment_nesting();
        torch::Tensor nested_result = tensor.pow(2);
        (void)nested_result.sum();
        at::autocast::decrement_nesting();
        at::autocast::decrement_nesting();
        
        // Try with different tensor types
        if (offset + 1 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            at::autocast::increment_nesting();
            torch::Tensor mixed_result = tensor + another_tensor;
            (void)mixed_result.sum();
            at::autocast::decrement_nesting();
        }
        
        // Try with extreme nesting levels
        uint8_t nesting_level = 1;
        if (offset < Size) {
            nesting_level = Data[offset++] % 10; // Limit to reasonable number
        }
        
        for (uint8_t i = 0; i < nesting_level; i++) {
            at::autocast::increment_nesting();
        }
        
        torch::Tensor deep_nested_result = tensor.sin();
        (void)deep_nested_result.sum();
        
        for (uint8_t i = 0; i < nesting_level; i++) {
            at::autocast::decrement_nesting();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
