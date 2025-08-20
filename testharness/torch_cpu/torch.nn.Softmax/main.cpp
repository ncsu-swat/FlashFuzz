#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to apply softmax along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // If tensor has dimensions, make sure dim is within valid range
        if (input.dim() > 0) {
            // Allow negative dimensions (PyTorch handles them by wrapping)
            dim = dim % (2 * input.dim()) - input.dim();
        }
        
        // Create Softmax module with the dimension
        torch::nn::Softmax softmax(dim);
        
        // Apply softmax to the input tensor
        torch::Tensor output = softmax(input);
        
        // Try different ways to call softmax
        torch::Tensor output2 = torch::nn::functional::softmax(input, torch::nn::functional::SoftmaxFuncOptions(dim));
        
        // Try with different dimensions if tensor has dimensions
        if (input.dim() > 0) {
            int64_t alt_dim = (dim + 1) % std::max(static_cast<int64_t>(1), static_cast<int64_t>(input.dim()));
            torch::nn::Softmax alt_softmax(alt_dim);
            torch::Tensor alt_output = alt_softmax(input);
        }
        
        // Try with default dimension (last dimension)
        if (input.dim() > 0) {
            torch::Tensor default_output = torch::nn::functional::softmax(input, torch::nn::functional::SoftmaxFuncOptions(-1));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}