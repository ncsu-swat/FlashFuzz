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
        
        // Create input tensor for tanh operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply tanh operation
        torch::Tensor output = torch::tanh(input);
        
        // Try in-place version if there's more data
        if (offset < Size && Data[offset] % 2 == 0) {
            torch::Tensor input_copy = input.clone();
            input_copy.tanh_();
        }
        
        // Try with different options if there's more data
        if (offset + 1 < Size) {
            // Create another tensor with remaining data
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Try tanh with out parameter
            torch::Tensor out = torch::empty_like(input2);
            torch::tanh_out(out, input2);
            
            // Try tanh with different tensor options
            if (offset + 1 < Size) {
                auto options = torch::TensorOptions()
                    .dtype(fuzzer_utils::parseDataType(Data[offset++]))
                    .requires_grad(Data[offset] % 2 == 0);
                
                torch::Tensor input3 = input.to(options);
                torch::Tensor output3 = torch::tanh(input3);
                
                // If requires_grad is true, try backward
                if (input3.requires_grad()) {
                    try {
                        output3.sum().backward();
                    } catch (...) {
                        // Ignore backward errors
                    }
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