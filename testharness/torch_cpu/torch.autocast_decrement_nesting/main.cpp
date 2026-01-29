#include "fuzzer_utils.h"       // General fuzzing utilities
#include <ATen/autocast_mode.h> // at::autocast helpers
#include <iostream>             // For cerr
#include <cstdint>

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
        
        // Extract parameters from fuzzer data
        uint8_t num_increments = 0;
        if (offset < Size) {
            // Limit to reasonable range to avoid too many increments
            num_increments = Data[offset++] % 5;
        }
        
        uint8_t num_decrements = 0;
        if (offset < Size) {
            num_decrements = Data[offset++] % 5;
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
        
        // Extract whether to set autocast enabled before operations
        bool set_enabled = false;
        if (offset < Size) {
            set_enabled = Data[offset++] & 0x1;
        }
        
        // Track how many times we actually increment so we can clean up
        int actual_increments = 0;
        
        // Increment nesting the requested number of times
        for (uint8_t i = 0; i < num_increments; i++) {
            at::autocast::increment_nesting();
            actual_increments++;
        }
        
        // Optionally set autocast state
        if (set_enabled) {
            at::autocast::set_autocast_enabled(device_type, true);
            at::autocast::set_autocast_dtype(device_type, dtype);
        }
        
        // Test tensor operations that might be affected by autocast state
        torch::Tensor result = tensor + tensor;
        
        // Use inner try-catch for operations that may fail due to shapes
        try {
            torch::Tensor matmul_result = torch::matmul(tensor, tensor);
        } catch (...) {
            // Shape mismatch is expected, silently ignore
        }
        
        // Call the autocast_decrement_nesting function - the main API being tested
        // Decrement up to the number requested, but track for cleanup
        int decrements_done = 0;
        for (uint8_t i = 0; i < num_decrements && i < actual_increments; i++) {
            at::autocast::decrement_nesting();
            decrements_done++;
        }
        
        // Test getting the current nesting level (if available)
        // This exercises related functionality
        
        // Clean up: ensure we decrement any remaining increments to restore state
        for (int i = decrements_done; i < actual_increments; i++) {
            at::autocast::decrement_nesting();
        }
        
        // Reset autocast state to avoid affecting future iterations
        if (set_enabled) {
            at::autocast::set_autocast_enabled(device_type, false);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}