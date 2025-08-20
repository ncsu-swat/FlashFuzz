#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte to select tensor type
        if (Size < 1) {
            return 0;
        }
        
        // Parse the tensor type selector from the first byte
        uint8_t type_selector = Data[0];
        offset++;
        
        // Map the selector to a tensor type
        std::vector<torch::TensorOptions> tensor_types = {
            torch::TensorOptions().dtype(torch::kFloat32),
            torch::TensorOptions().dtype(torch::kFloat64),
            torch::TensorOptions().dtype(torch::kInt32),
            torch::TensorOptions().dtype(torch::kInt64),
            torch::TensorOptions().dtype(torch::kInt16),
            torch::TensorOptions().dtype(torch::kInt8),
            torch::TensorOptions().dtype(torch::kUInt8),
            torch::TensorOptions().dtype(torch::kBool),
            torch::TensorOptions().dtype(torch::kHalf),
            torch::TensorOptions().dtype(torch::kBFloat16),
            torch::TensorOptions().dtype(torch::kComplexFloat),
            torch::TensorOptions().dtype(torch::kComplexDouble)
        };
        
        // Select a tensor type based on the selector
        torch::TensorOptions selected_type = tensor_types[type_selector % tensor_types.size()];
        
        // Set the default tensor type
        torch::set_default_dtype(selected_type.dtype());
        
        // Create a tensor to verify the default type was set correctly
        if (Size > offset) {
            // Create a tensor using the remaining data
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Create a tensor without specifying type - should use the default
            torch::Tensor default_tensor = torch::ones({2, 2});
            
            // Verify the default tensor has the expected type
            if (default_tensor.dtype() != selected_type.dtype()) {
                throw std::runtime_error("Default tensor type doesn't match the set type");
            }
            
            // Try some operations with the default tensor type
            torch::Tensor result = default_tensor + 1;
            torch::Tensor matmul_result = torch::matmul(default_tensor, default_tensor);
            
            // Reset to default float32
            torch::set_default_dtype(torch::kFloat32);
        } else {
            // Just test setting the default type without creating tensors
            torch::Tensor default_tensor = torch::ones({1});
            
            // Reset to default float32
            torch::set_default_dtype(torch::kFloat32);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}