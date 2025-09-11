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
        
        // Create input tensors for gammaincc(a, x)
        // gammaincc requires two tensors: a and x
        torch::Tensor a = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the torch.special.gammaincc operation
        torch::Tensor result = torch::special::gammaincc(a, x);
        
        // Optional: Test edge cases by creating additional tensors with specific properties
        if (offset + 4 < Size) {
            // Create a tensor with potentially extreme values
            torch::Tensor a_extreme = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor x_extreme = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test with extreme values
            torch::Tensor result_extreme = torch::special::gammaincc(a_extreme, x_extreme);
        }
        
        // Test scalar inputs by converting scalars to tensors
        if (a.dim() > 0 && x.dim() > 0) {
            // Extract scalars from tensors and convert to tensors
            torch::Scalar a_scalar = a.item();
            torch::Scalar x_scalar = x.item();
            
            // Convert scalars to tensors for testing
            torch::Tensor a_tensor_from_scalar = torch::tensor(a_scalar);
            torch::Tensor x_tensor_from_scalar = torch::tensor(x_scalar);
            
            // Test with scalar-derived tensors
            torch::Tensor result_scalar = torch::special::gammaincc(a_tensor_from_scalar, x);
            torch::Tensor result_scalar2 = torch::special::gammaincc(a, x_tensor_from_scalar);
            torch::Tensor result_scalar3 = torch::special::gammaincc(a_tensor_from_scalar, x_tensor_from_scalar);
        }
        
        // Test broadcasting with different shapes
        if (offset + 4 < Size) {
            torch::Tensor a_broadcast = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor x_broadcast = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test broadcasting
            torch::Tensor result_broadcast = torch::special::gammaincc(a_broadcast, x_broadcast);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
