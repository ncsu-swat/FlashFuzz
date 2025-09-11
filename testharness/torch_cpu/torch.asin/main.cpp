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
        
        // Create input tensor for asin operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply asin operation
        torch::Tensor result = torch::asin(input);
        
        // Try in-place version if there's more data
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            input_copy.asin_();
        }
        
        // Try with different options if there's more data
        if (offset + 1 < Size) {
            // Use the next byte to determine if we should try with different options
            uint8_t option_byte = Data[offset++];
            
            // Try with out tensor
            if (option_byte & 0x01) {
                torch::Tensor out = torch::empty_like(input);
                torch::asin_out(out, input);
            }
            
            // Try with different memory format if applicable
            if ((option_byte & 0x02) && input.dim() >= 4) {
                torch::Tensor channels_last = input.to(torch::MemoryFormat::ChannelsLast);
                torch::asin(channels_last);
            }
            
            // Try with different device if GPU is available
            if ((option_byte & 0x04) && torch::cuda::is_available()) {
                torch::Tensor cuda_input = input.to(torch::kCUDA);
                torch::asin(cuda_input);
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
