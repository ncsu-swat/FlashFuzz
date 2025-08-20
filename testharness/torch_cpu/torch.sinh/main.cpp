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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply sinh operation
        torch::Tensor result = torch::sinh(input);
        
        // Try different variants of the operation
        if (offset + 1 < Size) {
            // Use out variant if we have more data
            torch::Tensor out = torch::empty_like(input);
            torch::sinh_out(out, input);
            
            // Try in-place variant if tensor type supports it
            if (input.is_floating_point() || input.is_complex()) {
                torch::Tensor input_copy = input.clone();
                input_copy.sinh_();
            }
        }
        
        // Try with different options if we have more data
        if (offset + 2 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Try with different dtypes
            if (option_byte % 4 == 0) {
                torch::Tensor result_float = torch::sinh(input.to(torch::kFloat));
            } else if (option_byte % 4 == 1) {
                torch::Tensor result_double = torch::sinh(input.to(torch::kDouble));
            } else if (option_byte % 4 == 2 && torch::cuda::is_available()) {
                torch::Tensor result_cuda = torch::sinh(input.cuda());
            } else {
                torch::Tensor result_half = torch::sinh(input.to(torch::kHalf));
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