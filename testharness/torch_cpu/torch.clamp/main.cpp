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
        
        // Need at least some data to proceed
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract min and max values from the remaining data
        torch::Scalar min_val, max_val;
        
        // If we have enough data, extract min value
        if (offset + sizeof(float) <= Size) {
            float min_float;
            std::memcpy(&min_float, Data + offset, sizeof(float));
            offset += sizeof(float);
            min_val = torch::Scalar(min_float);
        } else {
            // Default min value if not enough data
            min_val = torch::Scalar(-1.0f);
        }
        
        // If we have enough data, extract max value
        if (offset + sizeof(float) <= Size) {
            float max_float;
            std::memcpy(&max_float, Data + offset, sizeof(float));
            offset += sizeof(float);
            max_val = torch::Scalar(max_float);
        } else {
            // Default max value if not enough data
            max_val = torch::Scalar(1.0f);
        }
        
        // Test different variants of clamp
        
        // Variant 1: clamp with both min and max
        torch::Tensor result1 = torch::clamp(input, min_val, max_val);
        
        // Variant 2: clamp with only min (max=None)
        torch::Tensor result2 = torch::clamp(input, min_val);
        
        // Variant 3: clamp_ (in-place) with both min and max
        torch::Tensor input_copy = input.clone();
        torch::Tensor result3 = torch::clamp_(input_copy, min_val, max_val);
        
        // Variant 4: clamp_ (in-place) with only min (max=None)
        input_copy = input.clone();
        torch::Tensor result4 = torch::clamp_(input_copy, min_val);
        
        // Variant 5: clamp with min=None, max only
        if (offset + sizeof(float) <= Size) {
            torch::Tensor result5 = torch::clamp(input, c10::nullopt, max_val);
            
            // Variant 6: clamp_ (in-place) with min=None, max only
            input_copy = input.clone();
            torch::Tensor result6 = torch::clamp_(input_copy, c10::nullopt, max_val);
        }
        
        // Test edge case: min > max
        if (min_val.toDouble() > max_val.toDouble()) {
            try {
                torch::Tensor edge_result = torch::clamp(input, min_val, max_val);
            } catch (const c10::Error& e) {
                // Expected exception when min > max
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
