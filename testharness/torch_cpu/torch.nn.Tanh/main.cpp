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
        
        // Create Tanh module
        torch::nn::Tanh tanh_module;
        
        // Apply Tanh operation
        torch::Tensor output = tanh_module->forward(input);
        
        // Alternative way to apply tanh
        torch::Tensor output2 = torch::tanh(input);
        
        // Try in-place version if possible
        if (input.is_floating_point()) {
            torch::Tensor input_copy = input.clone();
            input_copy.tanh_();
        }
        
        // Try with different options
        if (offset + 1 < Size) {
            bool train_mode = Data[offset++] % 2 == 0;
            tanh_module->train(train_mode);
            torch::Tensor output_train = tanh_module->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}