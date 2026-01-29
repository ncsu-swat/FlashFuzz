#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the neg operation
        torch::Tensor result = torch::neg(input_tensor);
        
        // Try in-place version
        try {
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.neg_();
        } catch (...) {
            // Silently ignore - in-place may fail for some tensor types
        }
        
        // Try with out parameter
        try {
            torch::Tensor out_tensor = torch::empty_like(input_tensor);
            torch::neg_out(out_tensor, input_tensor);
        } catch (...) {
            // Silently ignore
        }
        
        // Try with different data types
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            try {
                torch::Tensor converted_tensor = input_tensor.to(dtype);
                torch::Tensor result_with_dtype = torch::neg(converted_tensor);
            } catch (...) {
                // Silently ignore - conversion or neg may fail for some dtypes (e.g., bool)
            }
        }
        
        // Try with different memory formats
        if (offset < Size && input_tensor.dim() == 4) {
            uint8_t format_selector = Data[offset++];
            try {
                if (format_selector % 2 == 0) {
                    torch::Tensor channels_last = input_tensor.contiguous(torch::MemoryFormat::ChannelsLast);
                    torch::Tensor result_channels_last = torch::neg(channels_last);
                }
            } catch (...) {
                // Silently ignore - memory format conversion may fail
            }
        }
        
        // Try negation on a scalar tensor
        if (offset < Size) {
            try {
                torch::Tensor scalar = torch::tensor(static_cast<float>(Data[offset++]) - 128.0f);
                torch::Tensor neg_scalar = torch::neg(scalar);
            } catch (...) {
                // Silently ignore
            }
        }
        
        // Try with requires_grad for autograd path
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            try {
                torch::Tensor grad_tensor = input_tensor.to(torch::kFloat32).detach().requires_grad_(true);
                torch::Tensor grad_result = torch::neg(grad_tensor);
                grad_result.sum().backward();
            } catch (...) {
                // Silently ignore - autograd may fail
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