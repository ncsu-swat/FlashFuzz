#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 2 bytes for each tensor (dtype and rank)
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor if there's data left
        if (offset < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to perform matmul operation
            torch::Tensor result = torch::matmul(tensor1, tensor2);
            
            // Optional: do something with the result to ensure it's computed
            if (result.defined()) {
                volatile float sum = result.sum().item<float>();
            }
        } else {
            // If only one tensor was created, try matmul with itself
            torch::Tensor result = torch::matmul(tensor1, tensor1);
            
            // Optional: do something with the result to ensure it's computed
            if (result.defined()) {
                volatile float sum = result.sum().item<float>();
            }
        }
        
        // Try some edge cases if we have enough data
        if (Size > 8 && offset < Size - 4) {
            // Try with a 1D tensor
            std::vector<int64_t> shape1 = {3};
            torch::Tensor vec1 = torch::ones(shape1);
            
            // Try matmul with our fuzzed tensor
            try {
                torch::Tensor result = torch::matmul(vec1, tensor1);
            } catch (...) {
                // Ignore exceptions from this edge case
            }
            
            // Try matmul with transposed tensor
            try {
                if (tensor1.dim() >= 2) {
                    torch::Tensor transposed = tensor1.transpose(0, tensor1.dim()-1);
                    torch::Tensor result = torch::matmul(tensor1, transposed);
                }
            } catch (...) {
                // Ignore exceptions from this edge case
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