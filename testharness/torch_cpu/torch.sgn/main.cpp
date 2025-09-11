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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.sgn operation
        torch::Tensor result = torch::sgn(input);
        
        // Try different variants of the operation
        if (offset + 1 < Size) {
            // Use out variant if we have more data
            torch::Tensor out = torch::empty_like(input);
            torch::sgn_out(out, input);
            
            // Try in-place variant if tensor type supports it
            if (input.is_floating_point() || input.is_complex()) {
                torch::Tensor input_copy = input.clone();
                input_copy.sgn_();
            }
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Create a view if possible
            if (option_byte % 3 == 0 && input.numel() > 0) {
                torch::Tensor view = input.view({-1});
                torch::sgn(view);
            }
            
            // Try with non-contiguous tensor
            if (option_byte % 3 == 1 && input.dim() > 0 && input.size(0) > 1) {
                torch::Tensor non_contig = input.transpose(0, input.dim() - 1);
                if (!non_contig.is_contiguous()) {
                    torch::sgn(non_contig);
                }
            }
            
            // Try with strided tensor
            if (option_byte % 3 == 2 && input.dim() > 0 && input.size(0) > 1) {
                torch::Tensor strided = input.slice(0, 0, input.size(0), 2);
                torch::sgn(strided);
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
