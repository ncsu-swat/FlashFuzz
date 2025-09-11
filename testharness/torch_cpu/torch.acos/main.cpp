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
        
        // Create input tensor for acos operation
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply acos operation
        torch::Tensor result = torch::acos(input_tensor);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto accessor = result.accessor<float, 1>();
            volatile float first_element = accessor[0];
        }
        
        // Try some variants of the operation
        if (offset < Size) {
            // Create another tensor with remaining data if possible
            torch::Tensor input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test in-place version
            torch::Tensor inplace_result = input_tensor2.clone();
            inplace_result.acos_();
            
            // Test out variant
            torch::Tensor out_tensor = torch::empty_like(input_tensor2);
            torch::acos_out(out_tensor, input_tensor2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
