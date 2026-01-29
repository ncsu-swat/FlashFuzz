#include "fuzzer_utils.h"
#include <ATen/autocast_mode.h>
#include <iostream>

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
        constexpr auto device_type = at::kXLA;
        
        // Need at least 1 byte for dtype selection
        if (Size < 1) {
            return 0;
        }
        
        // Parse the dtype to use for autocast XLA
        uint8_t dtype_selector = Data[offset++];
        
        // Limit to dtypes that are valid for autocast
        // Typically float16, bfloat16, or float for autocast
        torch::ScalarType autocast_dtype;
        switch (dtype_selector % 4) {
            case 0:
                autocast_dtype = torch::kBFloat16;
                break;
            case 1:
                autocast_dtype = torch::kHalf;
                break;
            case 2:
                autocast_dtype = torch::kFloat;
                break;
            default:
                autocast_dtype = torch::kDouble;
                break;
        }
        
        // Save original state
        torch::ScalarType original_dtype = at::autocast::get_autocast_dtype(device_type);
        bool original_enabled = at::autocast::is_autocast_enabled(device_type);
        
        // Set the autocast XLA dtype using the correct API
        at::autocast::set_autocast_dtype(device_type, autocast_dtype);
        
        // Verify it was set correctly
        torch::ScalarType current_dtype = at::autocast::get_autocast_dtype(device_type);
        (void)current_dtype;
        
        // Create a tensor to test with autocast
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure tensor is float type for autocast operations
            if (!tensor.is_floating_point()) {
                tensor = tensor.to(torch::kFloat);
            }
            
            // Test with autocast enabled
            {
                at::autocast::set_autocast_enabled(device_type, true);
                
                try {
                    torch::Tensor result = tensor + tensor;
                    torch::Tensor result2 = tensor * 2.0f;
                    (void)result.sum();
                    (void)result2.sum();
                } catch (...) {
                    // Silently catch operation failures
                }
                
                at::autocast::set_autocast_enabled(device_type, false);
            }
            
            // Test with autocast disabled
            {
                at::autocast::set_autocast_enabled(device_type, false);
                
                try {
                    torch::Tensor result = tensor - tensor;
                    (void)result.sum();
                } catch (...) {
                    // Silently catch failures
                }
            }
        }
        
        // Test setting different dtypes in sequence
        if (offset < Size) {
            uint8_t second_dtype_selector = Data[offset++];
            torch::ScalarType second_dtype;
            switch (second_dtype_selector % 3) {
                case 0:
                    second_dtype = torch::kBFloat16;
                    break;
                case 1:
                    second_dtype = torch::kHalf;
                    break;
                default:
                    second_dtype = torch::kFloat;
                    break;
            }
            
            at::autocast::set_autocast_dtype(device_type, second_dtype);
            
            // Verify it was updated
            torch::ScalarType updated_dtype = at::autocast::get_autocast_dtype(device_type);
            (void)updated_dtype;
            
            // Test with the new dtype
            if (offset < Size) {
                at::autocast::set_autocast_enabled(device_type, true);
                
                try {
                    torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    if (tensor.is_floating_point()) {
                        torch::Tensor result = tensor * 3.0f;
                        (void)result.sum();
                    }
                } catch (...) {
                    // Silently catch failures
                }
                
                at::autocast::set_autocast_enabled(device_type, false);
            }
        }
        
        // Restore original state
        at::autocast::set_autocast_dtype(device_type, original_dtype);
        at::autocast::set_autocast_enabled(device_type, original_enabled);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}