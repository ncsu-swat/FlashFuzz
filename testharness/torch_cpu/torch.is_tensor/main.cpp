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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test tensor.is_tensor() method on the created tensor
        bool is_tensor_result = tensor.is_tensor();
        
        // Test with tensor reference
        torch::Tensor& tensor_ref = tensor;
        bool is_ref_tensor = tensor_ref.is_tensor();
        
        // Create a scalar tensor and test
        if (offset + 1 < Size) {
            torch::Tensor scalar_tensor = torch::scalar_tensor(static_cast<int>(Data[offset++]));
            bool is_scalar_tensor = scalar_tensor.is_tensor();
        }
        
        // Create another tensor and test
        if (offset < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            bool is_another_tensor = another_tensor.is_tensor();
        }
        
        // Test with cloned tensor
        torch::Tensor cloned_tensor = tensor.clone();
        bool is_cloned_tensor = cloned_tensor.is_tensor();
        
        // Test with detached tensor
        torch::Tensor detached_tensor = tensor.detach();
        bool is_detached_tensor = detached_tensor.is_tensor();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
