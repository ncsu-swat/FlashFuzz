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
        
        // Apply torch.special.gammaln operation
        torch::Tensor result = torch::special::gammaln(input);
        
        // Try some variants with options
        if (offset + 1 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Try with out tensor
            torch::Tensor out = torch::empty_like(input);
            torch::special::gammaln_out(out, input);
            
            // Try with non-standard memory layout if tensor has multiple dimensions
            if (input.dim() > 1) {
                torch::Tensor transposed = input.transpose(0, input.dim() - 1);
                torch::Tensor result_transposed = torch::special::gammaln(transposed);
            }
            
            // Try with different dtypes if we have enough data
            if (offset < Size) {
                auto dtype_selector = Data[offset++] % 2;
                if (dtype_selector == 0 && input.dtype() != torch::kDouble) {
                    torch::Tensor double_input = input.to(torch::kDouble);
                    torch::Tensor double_result = torch::special::gammaln(double_input);
                } else if (dtype_selector == 1 && input.dtype() != torch::kFloat) {
                    torch::Tensor float_input = input.to(torch::kFloat);
                    torch::Tensor float_result = torch::special::gammaln(float_input);
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
