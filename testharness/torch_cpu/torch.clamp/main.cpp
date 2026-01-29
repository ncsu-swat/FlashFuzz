#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
        float min_float = -1.0f;
        float max_float = 1.0f;
        
        // If we have enough data, extract min value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&min_float, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Handle NaN/Inf by replacing with defaults
            if (std::isnan(min_float) || std::isinf(min_float)) {
                min_float = -1.0f;
            }
        }
        
        // If we have enough data, extract max value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&max_float, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Handle NaN/Inf by replacing with defaults
            if (std::isnan(max_float) || std::isinf(max_float)) {
                max_float = 1.0f;
            }
        }
        
        // Ensure min <= max for valid clamping
        if (min_float > max_float) {
            std::swap(min_float, max_float);
        }
        
        torch::Scalar min_val(min_float);
        torch::Scalar max_val(max_float);
        
        // Determine which variant to test based on fuzzer data
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset] % 6;
            offset++;
        }
        
        switch (variant) {
            case 0: {
                // Variant 1: clamp with both min and max as scalars
                torch::Tensor result = torch::clamp(input, min_val, max_val);
                break;
            }
            case 1: {
                // Variant 2: clamp with only min (max=nullopt)
                torch::Tensor result = torch::clamp(input, min_val, c10::nullopt);
                break;
            }
            case 2: {
                // Variant 3: clamp with only max (min=nullopt)
                torch::Tensor result = torch::clamp(input, c10::nullopt, max_val);
                break;
            }
            case 3: {
                // Variant 4: clamp_ (in-place) with both min and max
                torch::Tensor input_copy = input.clone();
                input_copy.clamp_(min_val, max_val);
                break;
            }
            case 4: {
                // Variant 5: clamp with tensor min and max
                torch::Tensor min_tensor = torch::full_like(input, min_float);
                torch::Tensor max_tensor = torch::full_like(input, max_float);
                torch::Tensor result = torch::clamp(input, min_tensor, max_tensor);
                break;
            }
            case 5: {
                // Variant 6: clamp_min and clamp_max separately
                torch::Tensor result1 = torch::clamp_min(input, min_val);
                torch::Tensor result2 = torch::clamp_max(input, max_val);
                break;
            }
        }
        
        // Also test in-place variants with optional args
        if (offset < Size && (Data[offset] % 2 == 0)) {
            torch::Tensor input_copy = input.clone();
            input_copy.clamp_(c10::optional<torch::Scalar>(min_val), c10::optional<torch::Scalar>(max_val));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}