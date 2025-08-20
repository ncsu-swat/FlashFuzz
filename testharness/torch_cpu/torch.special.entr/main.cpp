#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.entr operation
        // entr(x) = -x * log(x) for x > 0, 0 for x == 0, and -inf for x < 0
        torch::Tensor result = torch::special::entr(input);
        
        // Try some edge cases with modified tensors
        if (offset + 1 < Size) {
            // Create a tensor with some negative values to test edge case
            torch::Tensor neg_input = input - Data[offset];
            torch::Tensor neg_result = torch::special::entr(neg_input);
            
            // Create a tensor with zeros to test another edge case
            torch::Tensor zero_input = torch::zeros_like(input);
            torch::Tensor zero_result = torch::special::entr(zero_input);
            
            // Create a tensor with very small positive values
            torch::Tensor small_input = input.abs() * 1e-10;
            torch::Tensor small_result = torch::special::entr(small_input);
            
            // Create a tensor with very large values
            torch::Tensor large_input = input.abs() * 1e10;
            torch::Tensor large_result = torch::special::entr(large_input);
            
            // Create a tensor with NaN values
            torch::Tensor nan_input = input.clone();
            nan_input.index_put_({input > 0}, std::numeric_limits<float>::quiet_NaN());
            torch::Tensor nan_result = torch::special::entr(nan_input);
            
            // Create a tensor with Inf values
            torch::Tensor inf_input = input.clone();
            inf_input.index_put_({input > 0}, std::numeric_limits<float>::infinity());
            torch::Tensor inf_result = torch::special::entr(inf_input);
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            // Create a non-contiguous tensor by slicing
            if (input.dim() > 0 && input.size(0) > 1) {
                torch::Tensor strided = input.slice(0, 0, input.size(0), 2);
                torch::Tensor strided_result = torch::special::entr(strided);
            }
            
            // Try with different dtype if possible
            if (input.scalar_type() != torch::kDouble) {
                torch::Tensor double_input = input.to(torch::kDouble);
                torch::Tensor double_result = torch::special::entr(double_input);
            } else if (input.scalar_type() != torch::kFloat) {
                torch::Tensor float_input = input.to(torch::kFloat);
                torch::Tensor float_result = torch::special::entr(float_input);
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