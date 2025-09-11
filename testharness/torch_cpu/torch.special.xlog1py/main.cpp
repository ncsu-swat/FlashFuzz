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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors for torch.special.xlog1py
        // This function requires two tensors: x and y
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            // Not enough data for second tensor, use a simple tensor instead
            torch::Tensor y = torch::ones_like(x);
            
            // Apply the operation
            torch::Tensor result = torch::special::xlog1py(x, y);
            
            // Force evaluation of the tensor
            result.sum().item<float>();
            
            return 0;
        }
        
        // Create the second tensor
        torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the operation
        // xlog1py(x, y) = x * log(1 + y) where x = 0 and y = -1 is defined as 0
        torch::Tensor result = torch::special::xlog1py(x, y);
        
        // Force evaluation of the tensor
        result.sum().item<float>();
        
        // Try broadcasting version if tensors have different shapes
        if (x.sizes() != y.sizes()) {
            // Try broadcasting in both directions
            try {
                torch::Tensor broadcast_result = torch::special::xlog1py(x, y);
                broadcast_result.sum().item<float>();
            } catch (...) {
                // Ignore errors from broadcasting
            }
            
            try {
                torch::Tensor broadcast_result2 = torch::special::xlog1py(y, x);
                broadcast_result2.sum().item<float>();
            } catch (...) {
                // Ignore errors from broadcasting
            }
        }
        
        // Try scalar versions if we have enough data
        if (offset + 1 < Size) {
            float scalar_value = static_cast<float>(Data[offset]) / 255.0f;
            
            // Try tensor-scalar operations
            try {
                torch::Tensor scalar_result1 = torch::special::xlog1py(x, scalar_value);
                scalar_result1.sum().item<float>();
            } catch (...) {
                // Ignore errors
            }
            
            try {
                torch::Tensor scalar_result2 = torch::special::xlog1py(scalar_value, y);
                scalar_result2.sum().item<float>();
            } catch (...) {
                // Ignore errors
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
