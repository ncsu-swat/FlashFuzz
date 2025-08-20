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
        
        // Apply torch.numel operation
        int64_t num_elements = tensor.numel();
        
        // Try to use the result to ensure it's not optimized away
        if (num_elements < 0) {
            throw std::runtime_error("Negative number of elements");
        }
        
        // Try alternative ways to call numel
        int64_t num_elements2 = torch::numel(tensor);
        
        // Verify both methods give the same result
        if (num_elements != num_elements2) {
            throw std::runtime_error("Inconsistent numel results");
        }
        
        // Test numel on a view of the tensor if possible
        if (tensor.dim() > 0 && tensor.size(0) > 0) {
            torch::Tensor view = tensor.slice(0, 0, tensor.size(0));
            int64_t view_elements = view.numel();
            
            // The view should have the same number of elements as the original tensor
            if (view_elements != num_elements) {
                throw std::runtime_error("View numel mismatch");
            }
        }
        
        // Test numel on a reshaped tensor if possible
        if (num_elements > 0) {
            // Reshape to a 1D tensor
            torch::Tensor reshaped = tensor.reshape({num_elements});
            int64_t reshaped_elements = reshaped.numel();
            
            // The reshaped tensor should have the same number of elements
            if (reshaped_elements != num_elements) {
                throw std::runtime_error("Reshape numel mismatch");
            }
        }
        
        // Test numel on a clone of the tensor
        torch::Tensor clone = tensor.clone();
        int64_t clone_elements = clone.numel();
        
        if (clone_elements != num_elements) {
            throw std::runtime_error("Clone numel mismatch");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}