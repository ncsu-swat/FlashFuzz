#include "fuzzer_utils.h"
#include <iostream>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }
    
    try
    {
        size_t offset = 0;
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.sin operation
        torch::Tensor result = torch::sin(input_tensor);
        
        // Try some variations if we have more data
        if (offset + 1 < Size) {
            // Try in-place version
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.sin_();
            
            // Try with out parameter
            torch::Tensor out_tensor = torch::empty_like(input_tensor);
            torch::sin_out(out_tensor, input_tensor);
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            uint8_t option_selector = Data[offset++];
            
            // Try with non-contiguous tensor
            if (option_selector % 4 == 0 && input_tensor.dim() > 1 && input_tensor.size(0) > 1) {
                torch::Tensor non_contiguous = input_tensor.transpose(0, input_tensor.dim() - 1);
                torch::Tensor result_non_contiguous = torch::sin(non_contiguous);
            }
            
            // Try with different device if available (skip CUDA in CPU-only fuzzing)
            // if (option_selector % 4 == 1 && torch::cuda::is_available()) {
            //     torch::Tensor cuda_tensor = input_tensor.cuda();
            //     torch::Tensor cuda_result = torch::sin(cuda_tensor);
            // }
            
            // Try with requires_grad
            if (option_selector % 4 == 2) {
                // Only floating point types support autograd
                if (input_tensor.is_floating_point()) {
                    torch::Tensor grad_tensor = input_tensor.clone().detach().requires_grad_(true);
                    torch::Tensor grad_result = torch::sin(grad_tensor);
                    
                    // Backward requires scalar - sum first
                    try {
                        grad_result.sum().backward();
                    } catch (...) {
                        // Silently ignore autograd failures
                    }
                }
            }
            
            // Try with different dtype if possible
            if (option_selector % 4 == 3 && offset < Size) {
                torch::ScalarType target_dtype = fuzzer_utils::parseDataType(Data[offset++]);
                try {
                    torch::Tensor converted = input_tensor.to(target_dtype);
                    torch::Tensor converted_result = torch::sin(converted);
                } catch (...) {
                    // Silently ignore dtype conversion failures
                }
            }
        }
        
        // Additional coverage: test with specific tensor shapes
        if (offset + 4 < Size) {
            // Test with 0-dim tensor (scalar)
            torch::Tensor scalar_tensor = torch::tensor(static_cast<float>(Data[offset++]) / 255.0f * 6.28f - 3.14f);
            torch::Tensor scalar_result = torch::sin(scalar_tensor);
            
            // Test with 1-dim tensor
            std::vector<float> vec_data;
            size_t vec_len = std::min(static_cast<size_t>(Data[offset++] % 16 + 1), Size - offset);
            for (size_t i = 0; i < vec_len && offset < Size; i++) {
                vec_data.push_back(static_cast<float>(Data[offset++]) / 255.0f * 6.28f - 3.14f);
            }
            if (!vec_data.empty()) {
                torch::Tensor vec_tensor = torch::tensor(vec_data);
                torch::Tensor vec_result = torch::sin(vec_tensor);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}