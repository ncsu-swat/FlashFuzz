#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/core/ivalue.h>
#include <iostream> // For cerr
#include <vector>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        int touched_flags = 0;
        
        // Create a tensor from the input data and test isTensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        c10::IValue tensor_iv(tensor);
        bool is_tensor_result = tensor_iv.isTensor();
        touched_flags += static_cast<int>(is_tensor_result);
        
        // Test with scalar IValue (should NOT be a tensor)
        if (offset < Size) {
            int64_t scalar_val = static_cast<int64_t>(Data[offset++]);
            c10::IValue scalar_iv(scalar_val);
            bool is_scalar_tensor = scalar_iv.isTensor();
            touched_flags += static_cast<int>(!is_scalar_tensor); // Should be false
        }
        
        // Test with double IValue (should NOT be a tensor)
        if (offset < Size) {
            double double_val = static_cast<double>(Data[offset++]) / 255.0;
            c10::IValue double_iv(double_val);
            bool is_double_tensor = double_iv.isTensor();
            touched_flags += static_cast<int>(!is_double_tensor); // Should be false
        }
        
        // Test with bool IValue (should NOT be a tensor)
        if (offset < Size) {
            bool bool_val = (Data[offset++] % 2) == 0;
            c10::IValue bool_iv(bool_val);
            bool is_bool_tensor = bool_iv.isTensor();
            touched_flags += static_cast<int>(!is_bool_tensor); // Should be false
        }
        
        // Test with string IValue (should NOT be a tensor)
        c10::IValue string_iv(std::string("test"));
        bool is_string_tensor = string_iv.isTensor();
        touched_flags += static_cast<int>(!is_string_tensor); // Should be false
        
        // Test with None IValue (should NOT be a tensor)
        c10::IValue none_iv;
        bool is_none_tensor = none_iv.isTensor();
        touched_flags += static_cast<int>(!is_none_tensor); // Should be false
        
        // Create a scalar tensor and test (should BE a tensor)
        if (offset < Size) {
            torch::Tensor scalar_tensor = torch::scalar_tensor(static_cast<int>(Data[offset++]));
            c10::IValue scalar_tensor_iv(scalar_tensor);
            bool is_scalar_tensor = scalar_tensor_iv.isTensor();
            touched_flags += static_cast<int>(is_scalar_tensor);
        }
        
        // Test with cloned tensor (should BE a tensor)
        torch::Tensor cloned_tensor = tensor.clone();
        c10::IValue cloned_iv(cloned_tensor);
        bool is_cloned_tensor = cloned_iv.isTensor();
        touched_flags += static_cast<int>(is_cloned_tensor);
        
        // Test with detached tensor (should BE a tensor)
        torch::Tensor detached_tensor = tensor.detach();
        c10::IValue detached_iv(detached_tensor);
        bool is_detached_tensor = detached_iv.isTensor();
        touched_flags += static_cast<int>(is_detached_tensor);
        
        // Test with list of tensors IValue (should NOT be a tensor, but contains tensors)
        std::vector<torch::Tensor> tensor_list = {tensor, cloned_tensor};
        c10::IValue list_iv(tensor_list);
        bool is_list_tensor = list_iv.isTensor();
        touched_flags += static_cast<int>(!is_list_tensor); // Should be false
        
        // But we can check if it's a tensor list
        bool is_tensor_list = list_iv.isTensorList();
        touched_flags += static_cast<int>(is_tensor_list);

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