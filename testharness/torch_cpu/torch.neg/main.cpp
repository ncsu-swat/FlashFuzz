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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the neg operation
        torch::Tensor result = torch::neg(input_tensor);
        
        // Try in-place version if there's more data
        if (offset < Size) {
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.neg_();
        }
        
        // Try with out parameter if there's more data
        if (offset < Size) {
            torch::Tensor out_tensor = torch::empty_like(input_tensor);
            torch::neg_out(out_tensor, input_tensor);
        }
        
        // Try with different data types if there's more data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            torch::Tensor converted_tensor = input_tensor.to(dtype);
            torch::Tensor result_with_dtype = torch::neg(converted_tensor);
        }
        
        // Try with different memory formats if there's more data
        if (offset < Size) {
            uint8_t format_selector = Data[offset++];
            if (format_selector % 2 == 0 && input_tensor.dim() >= 4) {
                torch::Tensor channels_last = input_tensor.to(torch::MemoryFormat::ChannelsLast);
                torch::Tensor result_channels_last = torch::neg(channels_last);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
