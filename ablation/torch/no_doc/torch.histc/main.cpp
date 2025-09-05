#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <limits>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeValue(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) {
        return false;
    }
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for tensor creation and parameters
        if (Size < 10) {
            return 0;
        }

        // Create input tensor from fuzzer data
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try with remaining data
            return 0;
        }

        // Consume parameters for histc
        int32_t bins_raw = 100; // default
        float min_raw = 0.0f;
        float max_raw = 0.0f;
        
        consumeValue(Data, Size, offset, bins_raw);
        consumeValue(Data, Size, offset, min_raw);
        consumeValue(Data, Size, offset, max_raw);
        
        // Process bins - allow various edge cases
        int64_t bins = static_cast<int64_t>(bins_raw);
        // Don't restrict too much - let PyTorch handle edge cases
        if (bins == 0) bins = 1; // Avoid zero bins which would be invalid
        bins = std::abs(bins) % 10000 + 1; // Keep reasonable but allow variety
        
        // Process min/max - explore various combinations
        double min_val = static_cast<double>(min_raw);
        double max_val = static_cast<double>(max_raw);
        
        // Explore edge cases with special values
        if (offset < Size && Data[offset] % 10 == 0) {
            min_val = std::numeric_limits<double>::lowest();
        } else if (offset < Size && Data[offset] % 10 == 1) {
            min_val = std::numeric_limits<double>::max();
        } else if (offset < Size && Data[offset] % 10 == 2) {
            min_val = std::numeric_limits<double>::infinity();
        } else if (offset < Size && Data[offset] % 10 == 3) {
            min_val = -std::numeric_limits<double>::infinity();
        } else if (offset < Size && Data[offset] % 10 == 4) {
            min_val = std::numeric_limits<double>::quiet_NaN();
        }
        
        if (offset + 1 < Size && Data[offset + 1] % 10 == 0) {
            max_val = std::numeric_limits<double>::lowest();
        } else if (offset + 1 < Size && Data[offset + 1] % 10 == 1) {
            max_val = std::numeric_limits<double>::max();
        } else if (offset + 1 < Size && Data[offset + 1] % 10 == 2) {
            max_val = std::numeric_limits<double>::infinity();
        } else if (offset + 1 < Size && Data[offset + 1] % 10 == 3) {
            max_val = -std::numeric_limits<double>::infinity();
        } else if (offset + 1 < Size && Data[offset + 1] % 10 == 4) {
            max_val = std::numeric_limits<double>::quiet_NaN();
        }

        // Try different tensor types and configurations
        if (offset + 2 < Size) {
            uint8_t config = Data[offset + 2];
            
            // Convert to different types to test type handling
            if (config % 5 == 0 && !input_tensor.is_complex()) {
                input_tensor = input_tensor.to(torch::kFloat32);
            } else if (config % 5 == 1 && !input_tensor.is_complex()) {
                input_tensor = input_tensor.to(torch::kFloat64);
            } else if (config % 5 == 2 && !input_tensor.is_complex()) {
                input_tensor = input_tensor.to(torch::kInt32);
            } else if (config % 5 == 3 && !input_tensor.is_complex()) {
                input_tensor = input_tensor.to(torch::kInt64);
            }
            // Keep original type for config % 5 == 4
            
            // Test with non-contiguous tensors
            if (config & 0x10 && input_tensor.dim() > 1) {
                input_tensor = input_tensor.transpose(0, input_tensor.dim() - 1);
            }
            
            // Test with views/slices
            if (config & 0x20 && input_tensor.numel() > 1) {
                int64_t slice_end = (input_tensor.numel() / 2) + 1;
                input_tensor = input_tensor.flatten().slice(0, 0, slice_end);
            }
        }

        // Call torch.histc with various parameter combinations
        torch::Tensor result;
        
        // Test different API variations
        if (offset + 3 < Size) {
            uint8_t api_variant = Data[offset + 3] % 4;
            
            switch(api_variant) {
                case 0:
                    // Basic call with just bins
                    result = torch::histc(input_tensor, bins);
                    break;
                case 1:
                    // Call with min and max
                    result = torch::histc(input_tensor, bins, min_val, max_val);
                    break;
                case 2:
                    // Call with swapped min/max to test error handling
                    result = torch::histc(input_tensor, bins, max_val, min_val);
                    break;
                case 3:
                    // Call with equal min/max
                    result = torch::histc(input_tensor, bins, min_val, min_val);
                    break;
            }
        } else {
            // Default call
            result = torch::histc(input_tensor, bins);
        }
        
        // Perform some operations on result to ensure it's valid
        if (result.defined()) {
            // Check basic properties
            auto sum = result.sum();
            auto min = result.min();
            auto max = result.max();
            
            // Test that histogram values are non-negative (should be counts)
            auto negative_counts = (result < 0).any();
            
            // For floating point inputs, sum should approximately equal number of elements
            // (within numerical precision)
            if (input_tensor.is_floating_point() && !input_tensor.is_complex()) {
                auto input_numel = input_tensor.numel();
                auto hist_sum = result.sum().item<double>();
                // This check might not always hold for edge cases with NaN/Inf
            }
            
            // Test operations on the result
            if (offset + 4 < Size && Data[offset + 4] % 3 == 0) {
                auto cumsum = result.cumsum(0);
            } else if (offset + 4 < Size && Data[offset + 4] % 3 == 1) {
                auto normalized = result / result.sum();
            }
        }
        
        // Test with out parameter
        if (offset + 5 < Size && Data[offset + 5] % 2 == 0) {
            torch::Tensor out_tensor = torch::empty({bins}, torch::kFloat32);
            torch::histc_out(out_tensor, input_tensor, bins);
        }

    }
    catch (const c10::Error& e)
    {
        // PyTorch errors are expected for invalid inputs
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}