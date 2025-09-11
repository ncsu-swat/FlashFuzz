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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a list of tensors
        std::vector<torch::Tensor> tensor_list;
        
        // Add the first tensor to the list
        tensor_list.push_back(input_tensor);
        
        // Try to create more tensors if we have data left
        for (int i = 0; i < 3 && offset < Size; i++) {
            try {
                torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensor_list.push_back(another_tensor);
            } catch (const std::exception &) {
                // If we can't create more tensors, just continue with what we have
                break;
            }
        }
        
        // Create a ModuleList to hold tensors
        torch::nn::ModuleList module_list;
        
        // Apply operations on the tensor list
        std::vector<torch::Tensor> result;
        for (const auto& tensor : tensor_list) {
            result.push_back(tensor);
        }
        
        // Try to access and use the result to ensure it's properly computed
        if (!result.empty()) {
            auto first_tensor = result[0];
            auto sum = first_tensor.sum();
            
            // Try to perform additional operations on the result
            if (result.size() > 1) {
                auto second_tensor = result[1];
                auto combined = first_tensor + second_tensor;
                auto product = combined * 0.5;
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
