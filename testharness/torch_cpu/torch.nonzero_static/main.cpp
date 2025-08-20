#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get size parameter from data
        uint8_t size_param = Data[offset++];
        int64_t size_value = static_cast<int64_t>(size_param) % 10 + 1;
        
        // Apply nonzero_static operation with required size parameter
        torch::Tensor result = torch::nonzero_static(input_tensor, size_value);
        
        // Try with custom fill_value if we have more data
        if (offset < Size) {
            uint8_t fill_param = Data[offset++];
            int64_t fill_value = static_cast<int64_t>(fill_param) - 128;
            
            torch::Tensor result_with_fill = torch::nonzero_static(input_tensor, size_value, fill_value);
        }
        
        // Try with different size values if we have more data
        if (offset < Size) {
            uint8_t size_param2 = Data[offset++];
            int64_t size_value2 = static_cast<int64_t>(size_param2) % 20 + 1;
            
            torch::Tensor result2 = torch::nonzero_static(input_tensor, size_value2);
        }
        
        // Try with both different size and fill_value if we have more data
        if (offset + 1 < Size) {
            uint8_t size_param3 = Data[offset++];
            uint8_t fill_param2 = Data[offset++];
            
            int64_t size_value3 = static_cast<int64_t>(size_param3) % 15 + 1;
            int64_t fill_value2 = static_cast<int64_t>(fill_param2) - 100;
            
            torch::Tensor result3 = torch::nonzero_static(input_tensor, size_value3, fill_value2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}