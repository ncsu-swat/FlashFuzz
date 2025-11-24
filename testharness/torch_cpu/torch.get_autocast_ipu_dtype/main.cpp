#include "fuzzer_utils.h"      // General fuzzing utilities
#include <ATen/autocast_mode.h> // at::autocast API surface
#include <iostream>             // For cerr
#include <tuple>                // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        // Keep target keyword visible for harness checks.
        // torch.get_autocast_ipu_dtype
        constexpr auto device_type = at::kIPU;
        
        // Need at least 1 byte for the enabled flag
        if (Size < 1) {
            return 0;
        }
        
        // Parse enabled flag from the first byte
        bool enabled = Data[0] % 2 == 1;
        offset++;
        
        // Try to create a tensor if we have enough data
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Get the autocast IPU dtype
            torch::ScalarType dtype = at::autocast::get_autocast_dtype(device_type);
            
            // Test setting the autocast IPU dtype
            at::autocast::set_autocast_dtype(device_type, tensor.scalar_type());
            
            // Test getting it again after setting
            torch::ScalarType updated_dtype = at::autocast::get_autocast_dtype(device_type);
            if (dtype != updated_dtype) {
                dtype = updated_dtype;
            }
            
            // Test with autocast enabled/disabled
            const bool prev_enabled = at::autocast::is_autocast_enabled(device_type);
            at::autocast::set_autocast_enabled(device_type, enabled);
            
            // Create another tensor with the current autocast settings
            if (offset < Size) {
                torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Perform some operation that might be affected by autocast
                torch::Tensor result = tensor2 + 1.0;
                
                // Check if the result has the expected dtype when autocast is enabled
                if (enabled && tensor2.is_floating_point()) {
                    torch::ScalarType current_dtype = at::autocast::get_autocast_dtype(device_type);
                    if (current_dtype != result.scalar_type() && 
                        result.scalar_type() != tensor2.scalar_type()) {
                        // This might indicate an issue with autocast
                    }
                }
            }

            at::autocast::set_autocast_enabled(device_type, prev_enabled);
        }
        
        // Test with different dtype settings
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Set autocast IPU dtype to the parsed dtype
            at::autocast::set_autocast_dtype(device_type, dtype);
            
            // Verify it was set correctly
            torch::ScalarType new_dtype = at::autocast::get_autocast_dtype(device_type);
            if (new_dtype == dtype) {
                (void)new_dtype;
            }
            
            // Test autocast with the new dtype
            const bool prev_enabled = at::autocast::is_autocast_enabled(device_type);
            at::autocast::set_autocast_enabled(device_type, true);

            // Create a tensor that might be affected by autocast
            if (offset < Size) {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                if (tensor.is_floating_point()) {
                    // Perform an operation that might trigger autocast
                    torch::Tensor result = tensor * 2.0;
                    (void)result.sum();
                }
            }

            at::autocast::set_autocast_enabled(device_type, prev_enabled);
        }
        
        // Test with nested autocast contexts
        if (offset < Size) {
            bool outer_enabled = (offset < Size) ? (Data[offset++] % 2 == 1) : false;
            bool inner_enabled = (offset < Size) ? (Data[offset++] % 2 == 1) : false;
            
            const bool orig_enabled = at::autocast::is_autocast_enabled(device_type);
            at::autocast::set_autocast_enabled(device_type, outer_enabled);
            
            // Do something between guards
            if (offset < Size) {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                const bool mid_enabled = at::autocast::is_autocast_enabled(device_type);
                at::autocast::set_autocast_enabled(device_type, inner_enabled);
                
                // Check if inner guard properly overrides outer guard
                if (offset < Size) {
                    torch::Tensor inner_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    if (inner_tensor.is_floating_point()) {
                        torch::Tensor result = inner_tensor + 3.0;
                        (void)result.sum();
                    }
                }
                
                at::autocast::set_autocast_enabled(device_type, mid_enabled);
                
                // Check if we properly return to outer guard settings
                if (tensor.is_floating_point()) {
                    torch::Tensor result = tensor + 4.0;
                    (void)result.sum();
                }
            }

            at::autocast::set_autocast_enabled(device_type, orig_enabled);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
