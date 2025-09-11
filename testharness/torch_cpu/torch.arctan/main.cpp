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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for arctan operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply arctan operation
        torch::Tensor result = torch::arctan(input);
        
        // Try in-place version if there's more data
        if (offset < Size && Size - offset > 0) {
            torch::Tensor input_copy = input.clone();
            input_copy.arctan_();
        }
        
        // Try with different output types if there's more data
        if (offset < Size && Size - offset > 0) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Create output tensor with different dtype
            torch::Tensor output = torch::empty_like(input, torch::TensorOptions().dtype(dtype));
            
            // Try arctan with output argument
            torch::arctan_out(output, input);
        }
        
        // Try arctan2 if we have more data to create a second tensor
        if (offset < Size && Size - offset > 2) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply arctan2 operation (atan2(input, input2))
            torch::Tensor result2 = torch::arctan2(input, input2);
            
            // Try in-place version of arctan2
            if (offset < Size && Size - offset > 0) {
                torch::Tensor input_copy = input.clone();
                input_copy.arctan2_(input2);
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
