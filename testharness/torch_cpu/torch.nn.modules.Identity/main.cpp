#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create an input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create an Identity module
        torch::nn::Identity identity_module;
        
        // Apply the Identity module to the input tensor
        torch::Tensor output_tensor = identity_module->forward(input_tensor);
        
        // Try with multiple tensors if we have enough data
        if (offset < Size) {
            torch::Tensor second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor output_second = identity_module->forward(second_tensor);
        }
        
        // Try with empty tensor
        torch::Tensor empty_tensor = torch::empty({0});
        torch::Tensor empty_output = identity_module->forward(empty_tensor);
        
        // Try with scalar tensor
        torch::Tensor scalar_tensor = torch::tensor(3.14);
        torch::Tensor scalar_output = identity_module->forward(scalar_tensor);
        
        // Try with boolean tensor
        torch::Tensor bool_tensor = torch::tensor(true);
        torch::Tensor bool_output = identity_module->forward(bool_tensor);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}