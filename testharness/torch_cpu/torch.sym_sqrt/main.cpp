#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply sqrt operation
        torch::Tensor result = torch::sqrt(input);
        
        // Try inplace version if there's more data
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            torch::sqrt_(input_copy);
        }
        
        // Try with different output dtype if there's more data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType output_dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Apply sqrt and convert to output dtype
            torch::Tensor result_with_dtype = torch::sqrt(input).to(output_dtype);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}