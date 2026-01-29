#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <limits>

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract min and max values for clipping if we have enough data
        double min_val = -10.0;
        double max_val = 10.0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&min_val, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Handle NaN - replace with default
            if (std::isnan(min_val)) min_val = -10.0;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_val, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Handle NaN - replace with default
            if (std::isnan(max_val)) max_val = 10.0;
        }
        
        // Ensure min <= max for valid clipping
        if (min_val > max_val) {
            std::swap(min_val, max_val);
        }
        
        // Variant 1: clip with both min and max using Scalars
        try {
            torch::Tensor result1 = torch::clip(input, 
                c10::optional<at::Scalar>(min_val), 
                c10::optional<at::Scalar>(max_val));
        } catch (const std::exception &) {
            // Shape or dtype issues - ignore
        }
        
        // Variant 2: clip with only min (max = nullopt)
        try {
            torch::Tensor result2 = torch::clip(input, 
                c10::optional<at::Scalar>(min_val), 
                c10::nullopt);
        } catch (const std::exception &) {
            // Ignore
        }
        
        // Variant 3: clip with only max (min = nullopt)
        try {
            torch::Tensor result3 = torch::clip(input, 
                c10::nullopt, 
                c10::optional<at::Scalar>(max_val));
        } catch (const std::exception &) {
            // Ignore
        }
        
        // Variant 4: in-place clipping using member function
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.clip_(
                c10::optional<at::Scalar>(min_val), 
                c10::optional<at::Scalar>(max_val));
        } catch (const std::exception &) {
            // Ignore
        }
        
        // Variant 5: clip with tensor min/max values if we have enough data
        if (offset + 4 < Size) {
            try {
                torch::Tensor min_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                if (offset + 4 < Size) {
                    torch::Tensor max_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Use the tensor overload
                    torch::Tensor result5 = torch::clip(input, 
                        c10::optional<at::Tensor>(min_tensor), 
                        c10::optional<at::Tensor>(max_tensor));
                }
            } catch (const std::exception &) {
                // Ignore exceptions from tensor creation or shape mismatches
            }
        }
        
        // Variant 6: clip with scalar tensor min only
        try {
            torch::Tensor min_tensor = torch::tensor(min_val);
            torch::Tensor result6 = torch::clip(input, 
                c10::optional<at::Tensor>(min_tensor), 
                c10::optional<at::Tensor>(c10::nullopt));
        } catch (const std::exception &) {
            // Ignore
        }
        
        // Variant 7: clip with scalar tensor max only
        try {
            torch::Tensor max_tensor = torch::tensor(max_val);
            torch::Tensor result7 = torch::clip(input, 
                c10::optional<at::Tensor>(c10::nullopt), 
                c10::optional<at::Tensor>(max_tensor));
        } catch (const std::exception &) {
            // Ignore
        }
        
        // Variant 8: clip with same min/max (produces constant tensor)
        try {
            double same_val = (min_val + max_val) / 2.0;
            torch::Tensor result8 = torch::clip(input, 
                c10::optional<at::Scalar>(same_val), 
                c10::optional<at::Scalar>(same_val));
        } catch (const std::exception &) {
            // Ignore
        }
        
        // Variant 9: clip with integer scalars
        try {
            int64_t int_min = static_cast<int64_t>(min_val);
            int64_t int_max = static_cast<int64_t>(max_val);
            torch::Tensor result9 = torch::clip(input, 
                c10::optional<at::Scalar>(int_min), 
                c10::optional<at::Scalar>(int_max));
        } catch (const std::exception &) {
            // Ignore
        }
        
        // Variant 10: clamp (alias for clip)
        try {
            torch::Tensor result10 = torch::clamp(input, 
                c10::optional<at::Scalar>(min_val), 
                c10::optional<at::Scalar>(max_val));
        } catch (const std::exception &) {
            // Ignore
        }
        
        // Variant 11: clamp_min
        try {
            torch::Tensor result11 = torch::clamp_min(input, min_val);
        } catch (const std::exception &) {
            // Ignore
        }
        
        // Variant 12: clamp_max
        try {
            torch::Tensor result12 = torch::clamp_max(input, max_val);
        } catch (const std::exception &) {
            // Ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}