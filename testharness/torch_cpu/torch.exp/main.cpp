#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.exp operation
        torch::Tensor result = torch::exp(input_tensor);
        
        // Try some variants of the operation
        if (offset + 1 < Size) {
            // Use out variant if we have more data
            torch::Tensor out_tensor = torch::empty_like(input_tensor);
            torch::exp_out(out_tensor, input_tensor);
            
            // Try in-place variant
            torch::Tensor inplace_tensor = input_tensor.clone();
            inplace_tensor.exp_();
        }
        
        // Try with different options if we have more data
        if (offset + 2 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Try with non-contiguous tensor
            if (option_byte & 0x01) {
                if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
                    torch::Tensor permuted = input_tensor.permute({input_tensor.dim()-1, 0});
                    torch::Tensor exp_result = torch::exp(permuted);
                }
            }
            
            // Try with different device if available
            if (option_byte & 0x02) {
                if (torch::cuda::is_available()) {
                    torch::Tensor cuda_tensor = input_tensor.cuda();
                    torch::Tensor cuda_result = torch::exp(cuda_tensor);
                }
            }
            
            // Try with different dtype
            if (option_byte & 0x04) {
                if (input_tensor.scalar_type() != torch::kDouble) {
                    torch::Tensor double_tensor = input_tensor.to(torch::kDouble);
                    torch::Tensor double_result = torch::exp(double_tensor);
                } else {
                    torch::Tensor float_tensor = input_tensor.to(torch::kFloat);
                    torch::Tensor float_result = torch::exp(float_tensor);
                }
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