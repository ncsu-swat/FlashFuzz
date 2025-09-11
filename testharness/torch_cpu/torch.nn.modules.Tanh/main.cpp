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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Tanh module
        torch::nn::Tanh tanh_module;
        
        // Apply Tanh operation
        torch::Tensor output = tanh_module->forward(input);
        
        // Alternative way to apply tanh
        torch::Tensor output2 = torch::tanh(input);
        
        // Try in-place version if supported
        if (input.is_floating_point()) {
            torch::Tensor input_copy = input.clone();
            input_copy.tanh_();
        }
        
        // Try with different options
        if (offset + 1 < Size) {
            bool inplace = Data[offset++] & 0x01;
            if (inplace && input.is_floating_point()) {
                torch::Tensor input_copy = input.clone();
                input_copy.tanh_();
            }
        }
        
        // Try with different tensor options
        if (offset + 1 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Try with contiguous tensor
            if (option_byte & 0x01) {
                torch::Tensor contiguous_input = input.contiguous();
                torch::Tensor contiguous_output = tanh_module->forward(contiguous_input);
            }
            
            // Try with non-contiguous tensor if possible
            if ((option_byte & 0x02) && input.dim() > 1 && input.size(0) > 1) {
                torch::Tensor transposed = input.transpose(0, input.dim() - 1);
                torch::Tensor transposed_output = tanh_module->forward(transposed);
            }
            
            // Try with different device if GPU is available
            if ((option_byte & 0x04) && torch::cuda::is_available()) {
                torch::Tensor cuda_input = input.to(torch::kCUDA);
                torch::Tensor cuda_output = tanh_module->forward(cuda_input);
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
