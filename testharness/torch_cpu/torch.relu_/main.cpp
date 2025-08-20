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
        
        // Make a copy of the original tensor for comparison
        torch::Tensor original = input_tensor.clone();
        
        // Apply relu_ in-place operation
        input_tensor.relu_();
        
        // Verify the operation worked correctly by comparing with manual relu
        torch::Tensor expected = torch::relu(original);
        
        // Check if the in-place operation produced the expected result
        if (!torch::allclose(input_tensor, expected)) {
            throw std::runtime_error("relu_ operation produced unexpected results");
        }
        
        // Try with different tensor options
        if (offset + 1 < Size) {
            // Create another tensor with remaining data
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Apply relu_ in-place
            another_tensor.relu_();
        }
        
        // Try with empty tensor
        torch::Tensor empty_tensor = torch::empty({0});
        empty_tensor.relu_();
        
        // Try with tensor containing extreme values
        if (Size > offset + 8) {
            std::vector<float> extreme_values = {
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::lowest(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN(),
                0.0f,
                -0.0f
            };
            
            torch::Tensor extreme_tensor = torch::tensor(extreme_values);
            extreme_tensor.relu_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}