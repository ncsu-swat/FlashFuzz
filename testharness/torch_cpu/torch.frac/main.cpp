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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.frac operation
        torch::Tensor result = torch::frac(input);
        
        // Try different variants of the operation
        if (offset + 1 < Size) {
            // Try in-place version if we have more data
            torch::Tensor input_copy = input.clone();
            input_copy.frac_();
            
            // Try out of place with options
            torch::TensorOptions options = torch::TensorOptions()
                .dtype(input.dtype())
                .device(input.device());
            torch::Tensor out = torch::empty_like(input, options);
            torch::frac_out(out, input);
        }
        
        // Try with different dtypes if we have more data
        if (offset + 2 < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Convert input to different dtype and apply frac
            torch::Tensor input_cast = input.to(dtype);
            torch::Tensor result_cast = torch::frac(input_cast);
        }
        
        // Try with non-contiguous tensor if we have more data
        if (offset + 1 < Size && input.dim() > 0 && input.numel() > 1) {
            // Create a non-contiguous view if possible
            torch::Tensor non_contiguous;
            if (input.dim() > 1 && input.size(0) > 1) {
                non_contiguous = input.transpose(0, input.dim() - 1);
                if (!non_contiguous.is_contiguous()) {
                    torch::Tensor result_non_contiguous = torch::frac(non_contiguous);
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
