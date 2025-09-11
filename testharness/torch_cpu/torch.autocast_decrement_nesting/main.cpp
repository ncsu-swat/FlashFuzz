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
        
        // Check if we have enough data to proceed
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to use in the context
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            tensor = torch::ones({2, 2});
        }
        
        // Extract a boolean from the data to determine if we should increment first
        bool increment_first = false;
        if (offset < Size) {
            increment_first = Data[offset++] & 0x1;
        }
        
        // Extract a device type from the data
        c10::DeviceType device_type = c10::kCPU;
        if (offset < Size) {
            uint8_t device_selector = Data[offset++];
            // Only use CUDA if available and selected
            if ((device_selector & 0x1) && torch::cuda::is_available()) {
                device_type = c10::kCUDA;
            }
        }
        
        // Extract a dtype from the data
        at::ScalarType dtype = at::kFloat;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            // Choose between common dtypes for autocast
            switch (dtype_selector % 4) {
                case 0: dtype = at::kFloat; break;
                case 1: dtype = at::kDouble; break;
                case 2: dtype = at::kHalf; break;
                case 3: dtype = at::kBFloat16; break;
            }
        }
        
        // If we're testing with increment first, increment the nesting level
        if (increment_first) {
            at::autocast::increment_nesting();
        }
        
        // Call the autocast_decrement_nesting function
        at::autocast::decrement_nesting();
        
        // Test a second decrement to see if it handles underflow correctly
        if (offset < Size && (Data[offset++] & 0x1)) {
            at::autocast::decrement_nesting();
        }
        
        // Test with a tensor operation to see if autocast state affects it
        torch::Tensor result = tensor + tensor;
        
        // Test setting the autocast state and then decrementing
        if (offset < Size && (Data[offset++] & 0x1)) {
            at::autocast::set_enabled(device_type, true);
            at::autocast::set_autocast_dtype(device_type, dtype);
            at::autocast::decrement_nesting();
        }
        
        // Test with a more complex operation that might be affected by autocast
        torch::Tensor complex_result = torch::matmul(tensor, tensor);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
