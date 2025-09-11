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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Mish module
        torch::nn::Mish mish_module;
        
        // Apply Mish operation
        torch::Tensor output = mish_module->forward(input);
        
        // Try functional version as well
        torch::Tensor output_functional = torch::mish(input);
        
        // Try inplace version if available
        if (input.is_floating_point()) {
            torch::Tensor input_copy = input.clone();
            torch::mish_(input_copy);
        }
        
        // Try with different options
        if (offset + 1 < Size) {
            bool inplace = Data[offset++] & 0x1;
            if (inplace && input.is_floating_point()) {
                torch::Tensor input_copy = input.clone();
                torch::mish_(input_copy);
            }
        }
        
        // Try with different tensor types
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Only try conversion if the tensor is valid
            if (input.numel() > 0 && input.defined()) {
                try {
                    torch::Tensor converted_input = input.to(dtype);
                    torch::Tensor converted_output = mish_module->forward(converted_input);
                } catch (const std::exception&) {
                    // Some dtype conversions might not be valid, that's fine
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
