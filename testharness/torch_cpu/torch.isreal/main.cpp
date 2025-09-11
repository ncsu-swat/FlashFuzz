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
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.isreal operation
        torch::Tensor result = torch::isreal(input_tensor);
        
        // Try to access the result to ensure computation is performed
        if (result.defined()) {
            auto numel = result.numel();
            
            // For non-empty tensors, access some values to ensure computation
            if (numel > 0) {
                auto first_val = result.item<bool>();
                
                // If tensor has more than one element, access the last one too
                if (numel > 1) {
                    auto flat = result.flatten();
                    auto last_val = flat[numel-1].item<bool>();
                }
            }
            
            // Test some edge cases with the result
            auto sum = result.sum();
            auto all_true = result.all().item<bool>();
            auto any_true = result.any().item<bool>();
        }
        
        // If we have more data, try another tensor with different properties
        if (offset + 2 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            torch::Tensor another_result = torch::isreal(another_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
