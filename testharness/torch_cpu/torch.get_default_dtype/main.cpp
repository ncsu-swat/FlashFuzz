#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Get the default dtype
        auto default_dtype = torch::get_default_dtype();
        
        // Try setting different dtypes and getting them back
        if (Size > 0) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Set the default dtype - convert ScalarType to TypeMeta
            torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(dtype));
            
            // Verify that get_default_dtype returns the dtype we just set
            auto new_default_dtype = torch::get_default_dtype();
            
            // Create a tensor with default dtype
            std::vector<int64_t> shape = {2, 3};
            auto tensor = torch::zeros(shape);
            
            // Verify the tensor has the expected dtype
            if (tensor.dtype() != new_default_dtype) {
                throw std::runtime_error("Tensor dtype doesn't match default dtype");
            }
            
            // Reset to original default dtype
            torch::set_default_dtype(default_dtype);
        }
        
        // Test with more complex tensor creation if we have enough data
        if (Size > offset) {
            // Create a tensor using the input data
            auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Get the default dtype again
            auto current_dtype = torch::get_default_dtype();
            
            // Create a new tensor with default dtype
            auto new_tensor = torch::zeros_like(tensor, torch::TensorOptions().dtype(current_dtype));
            
            // Verify the new tensor has the expected dtype
            if (new_tensor.dtype() != current_dtype) {
                throw std::runtime_error("New tensor dtype doesn't match current default dtype");
            }
        }
        
        // Test with different dtypes in sequence
        if (Size > offset + 2) {
            uint8_t dtype_selector1 = Data[offset++];
            uint8_t dtype_selector2 = Data[offset++];
            
            auto dtype1 = fuzzer_utils::parseDataType(dtype_selector1);
            auto dtype2 = fuzzer_utils::parseDataType(dtype_selector2);
            
            // Set first dtype - convert ScalarType to TypeMeta
            torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(dtype1));
            auto retrieved_dtype1 = torch::get_default_dtype();
            
            // Set second dtype - convert ScalarType to TypeMeta
            torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(dtype2));
            auto retrieved_dtype2 = torch::get_default_dtype();
            
            // Verify both retrievals match what was set
            if (retrieved_dtype1.toScalarType() != dtype1 || retrieved_dtype2.toScalarType() != dtype2) {
                throw std::runtime_error("Retrieved dtypes don't match set dtypes");
            }
            
            // Reset to original default dtype
            torch::set_default_dtype(default_dtype);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}