#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of the original tensor for comparison
        torch::Tensor original = tensor.clone();
        
        // Apply the cos_ operation in-place
        tensor.cos_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::cos(original);
        
        // Check if the results match
        if (tensor.sizes() != expected.sizes() || 
            tensor.dtype() != expected.dtype() ||
            !torch::allclose(tensor, expected, 1e-5, 1e-8)) {
            throw std::runtime_error("cos_ operation produced unexpected results");
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            // Create another tensor with different properties
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Apply cos_ to this tensor too
            tensor2.cos_();
        }
        
        // Test edge cases: empty tensor
        if (Size > offset + 1) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Create an empty tensor
            torch::Tensor empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(dtype));
            empty_tensor.cos_();
        }
        
        // Test with scalar tensor
        if (Size > offset + 1) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Create a scalar tensor
            torch::Tensor scalar_tensor;
            if (offset < Size) {
                scalar_tensor = torch::tensor(static_cast<float>(Data[offset]), 
                                             torch::TensorOptions().dtype(dtype));
            } else {
                scalar_tensor = torch::tensor(1.0, torch::TensorOptions().dtype(dtype));
            }
            
            scalar_tensor.cos_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}