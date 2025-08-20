#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse dimension parameter from the remaining data
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply Softmin operation
        // First, create the Softmin module
        torch::nn::Softmin softmin_module(torch::nn::SoftminOptions(dim));
        
        // Apply the Softmin operation to the input tensor
        torch::Tensor output = softmin_module->forward(input);
        
        // Alternative: use the functional interface
        torch::Tensor output_functional = torch::nn::functional::softmin(input, torch::nn::functional::SoftminFuncOptions().dim(dim));
        
        // Try with different dimensions if there's more data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t dim2;
            std::memcpy(&dim2, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Create another Softmin module with the new dimension
            torch::nn::Softmin softmin_module2(torch::nn::SoftminOptions(dim2));
            torch::Tensor output2 = softmin_module2->forward(input);
        }
        
        // Try with default dimension (last dimension)
        torch::nn::Softmin default_softmin = nullptr;
        if (input.dim() > 0) {
            default_softmin = torch::nn::Softmin(torch::nn::SoftminOptions(input.dim() - 1));
            torch::Tensor default_output = default_softmin->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}