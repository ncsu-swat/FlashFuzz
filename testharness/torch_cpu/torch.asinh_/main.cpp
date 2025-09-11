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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of the tensor for comparison
        torch::Tensor original = tensor.clone();
        
        // Apply the asinh_ operation in-place
        tensor.asinh_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::asinh(original);
        
        // Check if the operation produced the expected result
        if (tensor.sizes() != expected.sizes() || 
            tensor.dtype() != expected.dtype() ||
            !torch::allclose(tensor, expected, 1e-5, 1e-8)) {
            throw std::runtime_error("asinh_ operation produced unexpected result");
        }
        
        // Test with different tensor options if we have more data
        if (offset + 2 < Size) {
            // Create another tensor with different properties
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Make a copy for verification
            torch::Tensor original2 = tensor2.clone();
            
            // Apply asinh_ in-place
            tensor2.asinh_();
            
            // Verify with non-in-place version
            torch::Tensor expected2 = torch::asinh(original2);
            
            if (!torch::allclose(tensor2, expected2, 1e-5, 1e-8)) {
                throw std::runtime_error("asinh_ operation produced unexpected result on second tensor");
            }
        }
        
        // Test edge cases with special values if we have more data
        if (offset + 1 < Size) {
            // Create tensors with special values
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            
            // Create a tensor with special values: inf, -inf, NaN
            torch::Tensor special_values = torch::tensor(
                {std::numeric_limits<float>::infinity(), 
                 -std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::quiet_NaN(),
                 0.0f, -0.0f, 1.0f, -1.0f}, options);
            
            // Clone for verification
            torch::Tensor special_original = special_values.clone();
            
            // Apply asinh_ in-place
            special_values.asinh_();
            
            // Verify with non-in-place version (except for NaN which doesn't compare equal to itself)
            torch::Tensor special_expected = torch::asinh(special_original);
            
            // For elements that aren't NaN, verify they match
            auto special_mask = ~torch::isnan(special_original);
            if (!torch::allclose(
                    special_values.index({special_mask}), 
                    special_expected.index({special_mask}), 
                    1e-5, 1e-8)) {
                throw std::runtime_error("asinh_ operation produced unexpected result on special values");
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
