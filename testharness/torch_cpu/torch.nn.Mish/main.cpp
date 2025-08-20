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
        
        // Create Mish module
        torch::nn::Mish mish_module;
        
        // Apply Mish operation
        torch::Tensor output = mish_module->forward(input);
        
        // Alternative: use the functional version
        torch::Tensor output_functional = torch::nn::functional::mish(input);
        
        // Try with different tensor types
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Convert input to the new dtype if possible
            try {
                torch::Tensor input_converted = input.to(dtype);
                torch::Tensor output_converted = torch::nn::functional::mish(input_converted);
            } catch (const std::exception& e) {
                // Some dtypes might not be supported for Mish
            }
        }
        
        // Try with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor empty_output = mish_module->forward(empty_tensor);
        } catch (const std::exception& e) {
            // Empty tensor might not be supported
        }
        
        // Try with scalar tensor
        try {
            torch::Tensor scalar_tensor = torch::tensor(3.14);
            torch::Tensor scalar_output = mish_module->forward(scalar_tensor);
        } catch (const std::exception& e) {
            // Handle exception if scalar input is not supported
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}