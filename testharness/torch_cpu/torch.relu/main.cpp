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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply ReLU operation
        torch::Tensor output = torch::relu(input);
        
        // Try inplace version as well if there's enough data left
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            torch::relu_(input_copy);
        }
        
        // Try functional version with additional options if there's enough data left
        if (offset + 1 < Size) {
            bool inplace = Data[offset++] & 0x1;
            torch::Tensor input_copy = input.clone();
            torch::nn::functional::ReLUFuncOptions options;
            options.inplace(inplace);
            torch::Tensor output_func = torch::nn::functional::relu(input_copy, options);
        }
        
        // Try nn::ReLU module if there's enough data left
        if (offset < Size) {
            bool inplace = Data[offset++] & 0x1;
            torch::nn::ReLU relu_module(torch::nn::ReLUOptions().inplace(inplace));
            torch::Tensor input_copy = input.clone();
            torch::Tensor output_module = relu_module->forward(input_copy);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
