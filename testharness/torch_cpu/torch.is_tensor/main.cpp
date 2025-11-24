#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/core/ivalue.h>
#include <iostream> // For cerr
#include <tuple>    // For std::get with lu_unpack result

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
        
        // torch.is_tensor target keyword marker
        auto is_tensor_check = [](const torch::Tensor &t) {
            c10::IValue iv(t);
            return iv.isTensor();
        };

        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        int touched_flags = 0;
        
        // Test tensor with IValue::isTensor()
        bool is_tensor_result = is_tensor_check(tensor);
        touched_flags += static_cast<int>(is_tensor_result);
        
        // Test with tensor reference
        torch::Tensor& tensor_ref = tensor;
        bool is_ref_tensor = is_tensor_check(tensor_ref);
        touched_flags += static_cast<int>(is_ref_tensor);
        
        // Create a scalar tensor and test
        if (offset + 1 < Size) {
            torch::Tensor scalar_tensor = torch::scalar_tensor(static_cast<int>(Data[offset++]));
            bool is_scalar_tensor = is_tensor_check(scalar_tensor);
            touched_flags += static_cast<int>(is_scalar_tensor);
        }
        
        // Create another tensor and test
        if (offset < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            bool is_another_tensor = is_tensor_check(another_tensor);
            touched_flags += static_cast<int>(is_another_tensor);
        }
        
        // Test with cloned tensor
        torch::Tensor cloned_tensor = tensor.clone();
        bool is_cloned_tensor = is_tensor_check(cloned_tensor);
        touched_flags += static_cast<int>(is_cloned_tensor);
        
        // Test with detached tensor
        torch::Tensor detached_tensor = tensor.detach();
        bool is_detached_tensor = is_tensor_check(detached_tensor);
        touched_flags += static_cast<int>(is_detached_tensor);

        // Touch the combined result to avoid unused variable warnings
        if (touched_flags == -1) {
            std::cout << "Unreachable" << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
