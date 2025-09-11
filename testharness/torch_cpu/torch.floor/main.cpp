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
        
        // Apply floor operation
        torch::Tensor result = torch::floor(input_tensor);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto accessor = result.accessor<float, 1>();
            volatile float first_element = accessor[0];
            (void)first_element;
        }
        
        // Try floor_ (in-place version)
        if (input_tensor.is_floating_point()) {
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.floor_();
        }
        
        // Try floor with different output dtype
        if (input_tensor.is_floating_point()) {
            torch::Tensor result_int = torch::floor(input_tensor).to(torch::kInt);
        }
        
        // Try floor with non-contiguous tensor
        if (input_tensor.dim() > 1 && input_tensor.size(0) > 1) {
            torch::Tensor non_contiguous = input_tensor.transpose(0, input_tensor.dim() - 1);
            if (!non_contiguous.is_contiguous()) {
                torch::Tensor result_non_contiguous = torch::floor(non_contiguous);
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
