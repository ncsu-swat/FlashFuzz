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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.is_neg operation
        bool result = torch::is_neg(input_tensor);
        
        // Try alternative calling method
        bool result2 = input_tensor.is_neg();
        
        // Try with different tensor options
        if (offset + 1 < Size) {
            // Create a tensor with different options
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            bool result3 = torch::is_neg(tensor2);
        }
        
        // Try with empty tensor if we have enough data
        if (offset + 1 < Size) {
            torch::Tensor empty_tensor = torch::empty({0});
            bool empty_result = torch::is_neg(empty_tensor);
        }
        
        // Try with scalar tensor
        if (offset + 1 < Size) {
            torch::Tensor scalar_tensor = torch::tensor(Data[offset] % 2 == 0 ? 1 : -1);
            bool scalar_result = torch::is_neg(scalar_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
