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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.logical_not operation (equivalent to sym_not)
        torch::Tensor result = torch::logical_not(input_tensor);
        
        // Try to access the result to ensure computation is performed
        if (result.defined()) {
            auto sizes = result.sizes();
            auto dtype = result.dtype();
            
            // Force evaluation by accessing an element if tensor is not empty
            if (result.numel() > 0) {
                result.item();
            }
        }
        
        // Try with different input types if we have more data
        if (Size - offset > 2) {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor another_result = torch::logical_not(another_input);
            
            // Force evaluation
            if (another_result.defined() && another_result.numel() > 0) {
                another_result.item();
            }
        }
        
        // Try with boolean tensor specifically
        if (Size - offset > 2) {
            torch::Tensor bool_tensor = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kBool);
            torch::Tensor bool_result = torch::logical_not(bool_tensor);
            
            // Force evaluation
            if (bool_result.defined() && bool_result.numel() > 0) {
                bool_result.item();
            }
        }
        
        // Try with empty tensor
        torch::Tensor empty_tensor = torch::empty({0}, torch::kBool);
        torch::Tensor empty_result = torch::logical_not(empty_tensor);
        
        // Try with scalar tensor
        torch::Tensor scalar_tensor = torch::tensor(1, torch::kBool);
        torch::Tensor scalar_result = torch::logical_not(scalar_tensor);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
