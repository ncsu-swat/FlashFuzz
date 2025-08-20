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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make a copy of the original tensor for verification
        torch::Tensor original = tensor.clone();
        
        // Apply floor_ operation (in-place)
        tensor.floor_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::floor(original);
        
        // Check if the results match
        if (!torch::allclose(tensor, expected)) {
            std::cerr << "floor_ operation produced unexpected results" << std::endl;
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            // Create another tensor with different properties
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Apply floor_ operation
            tensor2.floor_();
        }
        
        // Try with empty tensor
        torch::Tensor empty_tensor = torch::empty({0});
        empty_tensor.floor_();
        
        // Try with scalar tensor
        if (offset < Size) {
            float value = static_cast<float>(Data[offset]);
            torch::Tensor scalar_tensor = torch::tensor(value);
            scalar_tensor.floor_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}