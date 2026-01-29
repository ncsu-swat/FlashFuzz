#include "fuzzer_utils.h"
#include <iostream>

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
        
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        // Get size parameter from data
        uint8_t size_param = Data[offset++];
        int64_t size_value = static_cast<int64_t>(size_param) % 100 + 1;
        
        // Apply nonzero_static operation with required size parameter
        try {
            torch::Tensor result = torch::nonzero_static(input_tensor, size_value);
        } catch (const std::exception &) {
            // Shape/size mismatch is expected for some inputs
        }
        
        // Try with custom fill_value if we have more data
        if (offset < Size) {
            uint8_t fill_param = Data[offset++];
            int64_t fill_value = static_cast<int64_t>(fill_param) - 128;
            
            try {
                torch::Tensor result_with_fill = torch::nonzero_static(input_tensor, size_value, fill_value);
            } catch (const std::exception &) {
                // Expected for some inputs
            }
        }
        
        // Try with different size values if we have more data
        if (offset < Size) {
            uint8_t size_param2 = Data[offset++];
            int64_t size_value2 = static_cast<int64_t>(size_param2) % 50 + 1;
            
            try {
                torch::Tensor result2 = torch::nonzero_static(input_tensor, size_value2);
            } catch (const std::exception &) {
                // Expected for some inputs
            }
        }
        
        // Try with both different size and fill_value if we have more data
        if (offset + 1 < Size) {
            uint8_t size_param3 = Data[offset++];
            uint8_t fill_param2 = Data[offset++];
            
            int64_t size_value3 = static_cast<int64_t>(size_param3) % 75 + 1;
            int64_t fill_value2 = static_cast<int64_t>(fill_param2) - 100;
            
            try {
                torch::Tensor result3 = torch::nonzero_static(input_tensor, size_value3, fill_value2);
            } catch (const std::exception &) {
                // Expected for some inputs
            }
        }
        
        // Test with zero-size tensor edge case
        if (offset < Size) {
            try {
                torch::Tensor empty_tensor = torch::empty({0});
                torch::Tensor result_empty = torch::nonzero_static(empty_tensor, 0);
            } catch (const std::exception &) {
                // Expected behavior
            }
        }
        
        // Test with boolean tensor (common use case for nonzero)
        if (offset < Size) {
            try {
                torch::Tensor bool_tensor = input_tensor.to(torch::kBool);
                uint8_t bool_size = Data[offset % Size];
                int64_t bool_size_val = static_cast<int64_t>(bool_size) % 50 + 1;
                torch::Tensor result_bool = torch::nonzero_static(bool_tensor, bool_size_val);
            } catch (const std::exception &) {
                // Expected for some inputs
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}