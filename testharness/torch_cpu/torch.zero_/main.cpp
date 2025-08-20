#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data to create a tensor
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the zero_ operation to the tensor
        tensor.zero_();
        
        // Verify that the tensor contains only zeros
        // This is not a sanity check but a verification that zero_ worked
        torch::Tensor zeros = torch::zeros_like(tensor);
        bool all_zeros = torch::all(tensor.eq(zeros)).item<bool>();
        
        // If not all zeros, something went wrong with the zero_ operation
        if (!all_zeros) {
            throw std::runtime_error("zero_ operation failed: tensor still contains non-zero values");
        }
        
        // Try to create another tensor if there's data left
        if (offset + 2 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            another_tensor.zero_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}