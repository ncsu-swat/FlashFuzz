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
        
        // Create input tensor for torch.erf
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.erf operation
        torch::Tensor result = torch::erf(input);
        
        // Try some variations of the API
        if (offset + 1 < Size) {
            // Use in-place version if we have more data
            torch::Tensor input_copy = input.clone();
            input_copy.erf_();
            
            // Try functional version with out parameter
            torch::Tensor out = torch::empty_like(input);
            torch::erf_out(out, input);
        }
        
        // Try with different device if we have more data
        if (offset + 1 < Size && torch::cuda::is_available()) {
            torch::Tensor cuda_input = input.to(torch::kCUDA);
            torch::Tensor cuda_result = torch::erf(cuda_input);
            
            // Try in-place on CUDA tensor
            cuda_input.erf_();
        }
        
        // Try with different dtypes if we have more data
        if (offset + 2 < Size) {
            // Try with float if not already
            if (input.dtype() != torch::kFloat) {
                torch::Tensor float_input = input.to(torch::kFloat);
                torch::Tensor float_result = torch::erf(float_input);
            }
            
            // Try with double if not already
            if (input.dtype() != torch::kDouble) {
                torch::Tensor double_input = input.to(torch::kDouble);
                torch::Tensor double_result = torch::erf(double_input);
            }
            
            // Try with half if not already
            if (input.dtype() != torch::kHalf) {
                torch::Tensor half_input = input.to(torch::kHalf);
                torch::Tensor half_result = torch::erf(half_input);
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