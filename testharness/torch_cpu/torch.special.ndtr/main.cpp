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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor for torch.special.ndtr
        // ndtr computes the cumulative distribution function of the standard normal distribution
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.ndtr operation
        torch::Tensor result = torch::special::ndtr(input);
        
        // Verify result is valid - use sum() to avoid .item() issues with multi-element tensors
        if (result.defined() && result.numel() > 0) {
            volatile float sum_val = result.sum().item<float>();
            (void)sum_val;
        }
        
        // Test with output tensor variant if we have more data
        if (Size - offset >= 4) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Pre-allocate output tensor with same shape
            torch::Tensor out = torch::empty_like(input2);
            
            // Use the out parameter variant
            torch::special::ndtr_out(out, input2);
            
            // Verify result
            if (out.defined() && out.numel() > 0) {
                volatile float sum_val2 = out.sum().item<float>();
                (void)sum_val2;
            }
        }
        
        // Test with specific dtypes to improve coverage
        if (Size - offset >= 2) {
            // Extract dtype choice from fuzzer data
            uint8_t dtype_choice = Data[offset % Size] % 3;
            offset++;
            
            torch::Tensor typed_input;
            try {
                torch::Tensor base_input = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Convert to different floating point types (ndtr requires floating point)
                switch (dtype_choice) {
                    case 0:
                        typed_input = base_input.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_input = base_input.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_input = base_input.to(torch::kFloat16);
                        break;
                    default:
                        typed_input = base_input.to(torch::kFloat32);
                        break;
                }
                
                torch::Tensor typed_result = torch::special::ndtr(typed_input);
                
                if (typed_result.defined() && typed_result.numel() > 0) {
                    volatile float sum_val3 = typed_result.sum().to(torch::kFloat32).item<float>();
                    (void)sum_val3;
                }
            } catch (const c10::Error &) {
                // Silently catch type conversion errors or unsupported dtype errors
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