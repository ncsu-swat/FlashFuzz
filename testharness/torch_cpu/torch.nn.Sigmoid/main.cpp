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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Sigmoid module
        torch::nn::Sigmoid sigmoid_module;
        
        // Apply Sigmoid operation
        torch::Tensor output = sigmoid_module->forward(input);
        
        // Alternative way to apply sigmoid
        torch::Tensor output2 = torch::sigmoid(input);
        
        // Try functional version as well
        torch::Tensor output3 = torch::sigmoid(input);
        
        // Try in-place version if possible
        if (input.is_floating_point()) {
            torch::Tensor input_copy = input.clone();
            input_copy.sigmoid_();
        }
        
        // Try with different options
        if (offset + 1 < Size) {
            bool inplace = Data[offset++] % 2 == 0;
            
            if (inplace && input.is_floating_point()) {
                torch::Tensor input_copy = input.clone();
                input_copy.sigmoid_();
            } else {
                torch::Tensor result = torch::sigmoid(input);
            }
        }
        
        // Try with gradients if possible
        if (input.is_floating_point() && offset + 1 < Size) {
            bool requires_grad = Data[offset++] % 2 == 0;
            
            if (requires_grad) {
                auto input_with_grad = input.clone().set_requires_grad(true);
                auto output_with_grad = torch::sigmoid(input_with_grad);
                
                // Try to compute gradients if the output is a scalar or we can sum it
                if (output_with_grad.numel() > 0) {
                    auto sum = output_with_grad.sum();
                    sum.backward();
                }
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