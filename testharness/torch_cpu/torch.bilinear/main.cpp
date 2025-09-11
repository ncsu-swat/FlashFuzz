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
        
        // Need at least 4 tensors for bilinear: input1, input2, weight, bias
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) return 0;
        
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) return 0;
        
        // Create weight tensor
        torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) return 0;
        
        // Create bias tensor
        torch::Tensor bias = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to apply bilinear operation
        // bilinear(input1, input2, weight, bias)
        // The bilinear function applies a bilinear transformation to the inputs
        torch::Tensor result;
        
        // Attempt to perform bilinear operation
        result = torch::bilinear(input1, input2, weight, bias);
        
        // Ensure the result is valid by performing a simple operation on it
        auto sum = result.sum();
        
        // Check if the result contains NaN or Inf values
        if (sum.isnan().any().item<bool>() || sum.isinf().any().item<bool>()) {
            // This is not an error, just a case we want to note
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
