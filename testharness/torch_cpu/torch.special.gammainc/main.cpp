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
        
        // Create input tensors for gammainc(a, x)
        // gammainc requires two input tensors: a and x
        torch::Tensor a = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the torch.special.gammainc operation
        // gammainc(a, x) computes the regularized lower incomplete gamma function
        torch::Tensor result = torch::special::gammainc(a, x);
        
        // Optional: Test edge cases by creating additional tensors with specific properties
        if (offset + 4 < Size) {
            // Create a tensor with potentially extreme values
            torch::Tensor a_extreme = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor x_extreme = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test with extreme values
            torch::Tensor result_extreme = torch::special::gammainc(a_extreme, x_extreme);
        }
        
        // Test scalar inputs if we have enough data
        if (offset + 2 < Size) {
            // Create scalar tensors
            torch::Tensor a_scalar = torch::tensor(static_cast<float>(Data[offset++]));
            torch::Tensor x_scalar = torch::tensor(static_cast<float>(Data[offset++]));
            
            // Test with scalar inputs
            torch::Tensor result_scalar = torch::special::gammainc(a_scalar, x_scalar);
        }
        
        // Test with broadcasting if we have enough data
        if (offset + 4 < Size) {
            // Create tensors with different shapes for broadcasting
            torch::Tensor a_broadcast = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor x_broadcast = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test with broadcasting
            torch::Tensor result_broadcast = torch::special::gammainc(a_broadcast, x_broadcast);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
