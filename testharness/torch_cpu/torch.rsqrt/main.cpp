#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor from the fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply rsqrt operation
        torch::Tensor result = torch::rsqrt(input);
        
        // Try in-place version if there's more data
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            input_copy.rsqrt_();
        }
        
        // Try with out parameter if there's more data
        if (offset < Size) {
            torch::Tensor out = torch::empty_like(input);
            torch::rsqrt_out(out, input);
        }
        
        // Try with different options if there's more data
        if (offset + 1 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Try with different dtypes if possible
            if (option_byte % 4 > 1) {
                try {
                    torch::ScalarType target_dtype = fuzzer_utils::parseDataType(Data[offset++]);
                    torch::Tensor result_cast = torch::rsqrt(input.to(target_dtype));
                } catch (...) {
                    // Ignore errors from dtype conversion
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