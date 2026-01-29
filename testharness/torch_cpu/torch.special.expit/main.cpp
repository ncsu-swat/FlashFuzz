#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>
#include <limits>

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for torch.special.expit
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.expit operation (sigmoid function)
        torch::Tensor result = torch::special::expit(input);
        
        // Try some edge cases with modified tensors if we have enough data
        if (offset + 1 < Size) {
            // Create a tensor with extreme values
            torch::Tensor extreme_input;
            
            uint8_t extreme_selector = Data[offset++];
            if (extreme_selector % 4 == 0) {
                // Very large positive values
                extreme_input = torch::full_like(input, 1e10);
            } else if (extreme_selector % 4 == 1) {
                // Very large negative values
                extreme_input = torch::full_like(input, -1e10);
            } else if (extreme_selector % 4 == 2) {
                // NaN values
                extreme_input = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
            } else {
                // Infinity values
                extreme_input = torch::full_like(input, std::numeric_limits<float>::infinity());
            }
            
            // Apply torch.special.expit to extreme values
            torch::Tensor extreme_result = torch::special::expit(extreme_input);
        }
        
        // Try with empty tensor if we have enough data
        if (offset + 1 < Size) {
            uint8_t empty_selector = Data[offset++];
            if (empty_selector % 2 == 0) {
                // Create an empty tensor with the same dtype as input
                std::vector<int64_t> empty_shape = {0};
                torch::Tensor empty_input = torch::empty(empty_shape, input.options());
                
                // Apply torch.special.expit to empty tensor
                torch::Tensor empty_result = torch::special::expit(empty_input);
            }
        }
        
        // Try with different data types if we have enough data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            dtype_selector = dtype_selector % 4;
            
            torch::ScalarType target_dtype;
            switch (dtype_selector) {
                case 0:
                    target_dtype = torch::kFloat;
                    break;
                case 1:
                    target_dtype = torch::kDouble;
                    break;
                case 2:
                    target_dtype = torch::kHalf;
                    break;
                default:
                    target_dtype = torch::kBFloat16;
                    break;
            }
            
            // Convert input to different dtype and apply expit
            try {
                torch::Tensor converted_input = input.to(target_dtype);
                torch::Tensor converted_result = torch::special::expit(converted_input);
            } catch (...) {
                // Some dtype conversions may fail, silently ignore
            }
        }
        
        // Try with a scalar tensor if we have enough data
        if (offset + 1 < Size) {
            uint8_t scalar_value = Data[offset++];
            torch::Tensor scalar_tensor = torch::tensor(static_cast<float>(scalar_value));
            torch::Tensor scalar_result = torch::special::expit(scalar_tensor);
        }
        
        // Try with negative infinity
        if (offset < Size) {
            torch::Tensor neg_inf_input = torch::full_like(input, -std::numeric_limits<float>::infinity());
            torch::Tensor neg_inf_result = torch::special::expit(neg_inf_input);
        }
        
        // Try with out parameter variant if available
        if (offset < Size) {
            torch::Tensor out_tensor = torch::empty_like(input);
            torch::special::expit_out(out_tensor, input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}