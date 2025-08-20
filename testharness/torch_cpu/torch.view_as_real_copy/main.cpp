#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if the tensor is complex
        bool is_complex = input_tensor.is_complex();
        
        // If the tensor is not complex, convert it to a complex tensor
        if (!is_complex) {
            // Create a complex tensor from the real tensor
            if (input_tensor.dtype() == torch::kFloat) {
                input_tensor = input_tensor.to(torch::kComplexFloat);
            } else if (input_tensor.dtype() == torch::kDouble) {
                input_tensor = input_tensor.to(torch::kComplexDouble);
            } else {
                // For other dtypes, convert to float first, then to complex
                input_tensor = input_tensor.to(torch::kFloat).to(torch::kComplexFloat);
            }
        }
        
        // Apply view_as_real_copy operation
        torch::Tensor result = torch::view_as_real_copy(input_tensor);
        
        // Perform some operations on the result to ensure it's used
        if (result.numel() > 0) {
            auto sum = result.sum();
            
            // Try to access the last dimension which should be 2 (real and imaginary parts)
            if (result.dim() > 0) {
                auto last_dim_size = result.size(-1);
                if (last_dim_size > 0) {
                    auto first_element = result.index({0});
                }
            }
        }
        
        // Try some edge cases if we have more data
        if (offset + 1 < Size) {
            // Create another tensor and try view_as_real_copy on it
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure it's complex
            if (!another_tensor.is_complex()) {
                if (another_tensor.dtype() == torch::kFloat) {
                    another_tensor = another_tensor.to(torch::kComplexFloat);
                } else if (another_tensor.dtype() == torch::kDouble) {
                    another_tensor = another_tensor.to(torch::kComplexDouble);
                } else {
                    another_tensor = another_tensor.to(torch::kFloat).to(torch::kComplexFloat);
                }
            }
            
            // Try with non-contiguous tensor
            if (another_tensor.dim() > 1 && another_tensor.size(0) > 1) {
                auto non_contiguous = another_tensor.transpose(0, another_tensor.dim() - 1);
                auto result_non_contiguous = torch::view_as_real_copy(non_contiguous);
            }
            
            // Try with zero-sized tensor if possible
            std::vector<int64_t> zero_shape = another_tensor.sizes().vec();
            if (!zero_shape.empty()) {
                zero_shape[0] = 0;
                torch::Tensor zero_tensor;
                if (another_tensor.dtype() == torch::kComplexFloat) {
                    zero_tensor = torch::empty(zero_shape, torch::kComplexFloat);
                } else {
                    zero_tensor = torch::empty(zero_shape, torch::kComplexDouble);
                }
                auto zero_result = torch::view_as_real_copy(zero_tensor);
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