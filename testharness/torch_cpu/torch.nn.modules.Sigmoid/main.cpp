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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for Sigmoid
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Sigmoid module
        torch::nn::Sigmoid sigmoid_module;
        
        // Apply Sigmoid operation
        torch::Tensor output = sigmoid_module->forward(input);
        
        // Alternative way to apply sigmoid
        torch::Tensor output2 = torch::sigmoid(input);
        
        // Try functional interface as well
        torch::Tensor output3 = torch::sigmoid(input);
        
        // Try in-place version if possible
        if (input.is_floating_point()) {
            torch::Tensor input_clone = input.clone();
            input_clone.sigmoid_();
        }
        
        // Try with different options
        if (offset + 1 < Size) {
            bool inplace = Data[offset++] % 2 == 0;
            
            // Create another tensor for additional tests
            if (offset < Size) {
                torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Apply sigmoid with different options
                torch::Tensor another_output = sigmoid_module->forward(another_input);
                
                // Try with gradients if possible
                if (another_input.is_floating_point()) {
                    another_input = another_input.detach().requires_grad_(true);
                    torch::Tensor grad_output = sigmoid_module->forward(another_input);
                    
                    // Try backward if tensor has non-zero elements
                    if (grad_output.numel() > 0) {
                        try {
                            grad_output.sum().backward();
                        } catch (...) {
                            // Ignore backward errors
                        }
                    }
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
