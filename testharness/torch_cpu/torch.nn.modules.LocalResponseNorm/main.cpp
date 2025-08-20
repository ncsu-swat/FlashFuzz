#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for LocalResponseNorm from the remaining data
        if (offset + 4 > Size) {
            return 0;
        }
        
        // Parse size parameter (must be positive)
        int64_t size = 1 + (Data[offset++] % 7); // Size between 1 and 7
        
        // Parse alpha parameter
        float alpha = 0.0001f;
        if (offset < Size) {
            uint8_t alpha_byte = Data[offset++];
            alpha = static_cast<float>(alpha_byte) / 255.0f; // Normalize to [0, 1]
            alpha = std::max(0.0001f, alpha); // Ensure it's positive
        }
        
        // Parse beta parameter
        float beta = 0.75f;
        if (offset < Size) {
            uint8_t beta_byte = Data[offset++];
            beta = static_cast<float>(beta_byte) / 255.0f * 2.0f; // Normalize to [0, 2]
            beta = std::max(0.01f, beta); // Ensure it's positive
        }
        
        // Parse k parameter
        float k = 1.0f;
        if (offset < Size) {
            uint8_t k_byte = Data[offset++];
            k = static_cast<float>(k_byte) / 255.0f * 2.0f; // Normalize to [0, 2]
        }
        
        // Create LocalResponseNorm module
        torch::nn::LocalResponseNorm lrn(
            torch::nn::LocalResponseNormOptions(size)
                .alpha(alpha)
                .beta(beta)
                .k(k)
        );
        
        // Apply the operation
        torch::Tensor output = lrn->forward(input);
        
        // Try with different input shapes if there's more data
        if (offset + 4 < Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor output2 = lrn->forward(input2);
        }
        
        // Try with edge case inputs if possible
        if (input.dim() > 0 && input.size(0) > 0) {
            // Test with a slice of the input
            torch::Tensor slice = input.slice(0, 0, input.size(0) / 2 + 1);
            torch::Tensor output_slice = lrn->forward(slice);
        }
        
        // Test with different data types if possible
        if (offset + 4 < Size) {
            // Try to create a tensor with a different dtype
            torch::Tensor input_float = input.to(torch::kFloat);
            torch::Tensor output_float = lrn->forward(input_float);
            
            if (offset + 4 < Size) {
                torch::Tensor input_double = input.to(torch::kDouble);
                torch::Tensor output_double = lrn->forward(input_double);
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