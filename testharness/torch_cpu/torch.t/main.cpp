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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.t() operation
        // torch.t() is a matrix transpose operation that requires a 2D tensor
        // For non-2D tensors, we'll still call it to test error handling
        torch::Tensor output_tensor = input_tensor.t();
        
        // Verify the operation worked correctly
        if (input_tensor.dim() == 2) {
            // For 2D tensors, dimensions should be swapped
            auto input_sizes = input_tensor.sizes();
            auto output_sizes = output_tensor.sizes();
            
            if (input_sizes[0] != output_sizes[1] || input_sizes[1] != output_sizes[0]) {
                throw std::runtime_error("Transpose operation failed: dimensions not properly swapped");
            }
        }
        
        // Try to create another tensor if we have more data
        if (offset + 2 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to transpose it too
            torch::Tensor another_output = another_tensor.t();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}