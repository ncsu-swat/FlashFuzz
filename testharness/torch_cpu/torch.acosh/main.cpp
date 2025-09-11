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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply acosh operation
        torch::Tensor result = torch::acosh(input);
        
        // Try in-place version if there's more data
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            input_copy.acosh_();
        }
        
        // Try with options if there's more data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            torch::Tensor result_with_dtype = torch::acosh(input.to(dtype));
            
            // Try out-of-place with named output
            torch::Tensor output = torch::empty_like(input, torch::TensorOptions().dtype(dtype));
            torch::acosh_out(output, input);
        }
        
        // Try with different memory formats if there's more data
        if (offset < Size) {
            uint8_t format_selector = Data[offset++];
            
            // Try different memory formats
            if (format_selector % 3 == 0 && input.dim() >= 4) {
                auto channels_last_input = input.to(torch::MemoryFormat::ChannelsLast);
                torch::Tensor channels_last_result = torch::acosh(channels_last_input);
            } else if (format_selector % 3 == 1 && input.dim() >= 5) {
                auto channels_last_3d_input = input.to(torch::MemoryFormat::ChannelsLast3d);
                torch::Tensor channels_last_3d_result = torch::acosh(channels_last_3d_input);
            }
        }
        
        // Try with non-contiguous tensor if there's more data
        if (offset < Size && input.dim() > 0 && input.numel() > 1) {
            // Create a non-contiguous view if possible
            torch::Tensor non_contiguous;
            if (input.dim() > 1 && input.size(0) > 1) {
                non_contiguous = input.slice(0, 0, input.size(0), 2);
                if (!non_contiguous.is_contiguous()) {
                    torch::Tensor non_contiguous_result = torch::acosh(non_contiguous);
                }
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
