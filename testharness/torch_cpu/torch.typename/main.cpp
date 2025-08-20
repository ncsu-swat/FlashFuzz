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
        
        // Create a tensor with various properties
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get scalar type and dtype information
        auto scalar_type = tensor.scalar_type();
        auto dtype = tensor.dtype();
        
        // Try to get the type name from the tensor options
        auto options = tensor.options();
        auto options_dtype = options.dtype();
        
        // Try to get the type name from the tensor's scalar type
        std::string scalar_type_name = torch::toString(scalar_type);
        
        // Try to get the type name from the tensor's dtype
        std::string dtype_name = torch::toString(dtype);
        
        // Try to get the type name from the tensor's options dtype
        std::string options_dtype_name = torch::toString(options_dtype);
        
        // Try to create a new tensor with the same type
        torch::Tensor new_tensor = torch::empty({1}, tensor.options());
        
        // Try to get the type name from the new tensor
        std::string new_tensor_dtype_name = torch::toString(new_tensor.dtype());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}