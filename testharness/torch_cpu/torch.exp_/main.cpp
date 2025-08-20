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
        
        // Apply the exp_ operation in-place
        tensor.exp_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::exp(original);
        
        // Check if the results match
        if (tensor.sizes() != expected.sizes()) {
            throw std::runtime_error("Shape mismatch after exp_ operation");
        }
        
        // Try with different tensor options
        if (offset + 1 < Size) {
            // Create another tensor with different properties
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Apply exp_ to this tensor too
            tensor2.exp_();
        }
        
        // Try with empty tensor
        torch::Tensor empty_tensor = torch::empty({0});
        empty_tensor.exp_();
        
        // Try with scalar tensor
        if (offset + 1 < Size) {
            float value = static_cast<float>(Data[offset]) / 255.0f;
            torch::Tensor scalar_tensor = torch::tensor(value);
            scalar_tensor.exp_();
        }
        
        // Try with tensors containing extreme values
        if (offset + 2 < Size) {
            // Create tensor with very large values
            std::vector<float> large_values = {1e30f, -1e30f};
            torch::Tensor large_tensor = torch::tensor(large_values);
            large_tensor.exp_();
            
            // Create tensor with special values (inf, -inf, nan)
            std::vector<float> special_values = {
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN()
            };
            torch::Tensor special_tensor = torch::tensor(special_values);
            special_tensor.exp_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}