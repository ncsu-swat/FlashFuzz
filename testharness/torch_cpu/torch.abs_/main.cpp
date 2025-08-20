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
        
        // Apply the abs_ operation in-place
        tensor.abs_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::abs(original);
        
        // Check if the results match
        if (!torch::allclose(tensor, expected)) {
            throw std::runtime_error("abs_ operation produced unexpected results");
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            // Create another tensor with different properties
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Apply abs_ to this tensor too
            tensor2.abs_();
            
            // Test on a view of the tensor if possible
            if (tensor2.numel() > 1 && tensor2.dim() > 0) {
                auto view = tensor2.slice(0, 0, tensor2.size(0) / 2 + 1);
                auto view_clone = view.clone();
                view.abs_();
                
                // Verify the view operation
                auto expected_view = torch::abs(view_clone);
                if (!torch::allclose(view, expected_view)) {
                    throw std::runtime_error("abs_ on tensor view produced unexpected results");
                }
            }
        }
        
        // Test with empty tensor if we have more data
        if (offset + 1 < Size) {
            torch::Tensor empty_tensor = torch::empty({0});
            empty_tensor.abs_();
        }
        
        // Test with scalar tensor
        if (offset + 1 < Size) {
            torch::Tensor scalar_tensor = torch::tensor(Data[offset] % 256 - 128);
            torch::Tensor scalar_copy = scalar_tensor.clone();
            scalar_tensor.abs_();
            
            if (scalar_tensor.item<int>() != std::abs(scalar_copy.item<int>())) {
                throw std::runtime_error("abs_ on scalar tensor produced unexpected results");
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