#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse dim parameter from the remaining data
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create Softmin module
        torch::nn::Softmin softmin_module(torch::nn::SoftminOptions(dim));
        
        // Apply Softmin to the input tensor
        torch::Tensor output = softmin_module->forward(input);
        
        // Try with different dimensions if there's more data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            torch::nn::Softmin softmin_module2(torch::nn::SoftminOptions(dim));
            torch::Tensor output2 = softmin_module2->forward(input);
        }
        
        // Try with default dimension (-1)
        torch::nn::Softmin default_softmin(torch::nn::SoftminOptions(-1));
        torch::Tensor default_output = default_softmin->forward(input);
        
        // Try with named parameters
        torch::nn::SoftminOptions options(dim);
        torch::nn::Softmin named_softmin(options);
        torch::Tensor named_output = named_softmin->forward(input);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}