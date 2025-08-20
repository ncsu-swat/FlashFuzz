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
        
        // Create a copy of the original tensor for comparison
        torch::Tensor original = input_tensor.clone();
        
        // Apply the square_ operation (in-place)
        input_tensor.square_();
        
        // Verify the operation worked correctly by comparing with manual squaring
        torch::Tensor expected = original * original;
        
        // Check if the results match
        if (input_tensor.sizes() != expected.sizes() || 
            !torch::allclose(input_tensor, expected, 1e-5, 1e-8)) {
            throw std::runtime_error("square_ operation produced unexpected results");
        }
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}