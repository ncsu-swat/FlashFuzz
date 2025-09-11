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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make a copy of the original tensor for comparison
        torch::Tensor original = input_tensor.clone();
        
        // Apply ceil_ operation (in-place)
        input_tensor.ceil_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::ceil(original);
        
        // Check if the in-place operation produced the same result as the non-in-place version
        if (!torch::allclose(input_tensor, expected)) {
            throw std::runtime_error("ceil_ produced different results than ceil");
        }
        
        // Try another approach with a view
        if (offset + 1 < Size && input_tensor.numel() > 0) {
            // Create a view if possible
            torch::Tensor view;
            if (input_tensor.dim() > 0) {
                view = input_tensor.slice(0, 0, input_tensor.size(0));
            } else {
                view = input_tensor;
            }
            
            // Apply ceil_ to the view
            view.ceil_();
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            // Create another tensor with different properties
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Apply ceil_ operation
            another_tensor.ceil_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
