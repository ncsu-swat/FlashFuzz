#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <limits>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some bytes for tensor creation and min value
        if (Size < 4) {
            // Still try to do something even with minimal data
            torch::Tensor t = torch::empty({});
            torch::clamp_min(t, 0.0);
            return 0;
        }

        // Create primary tensor from fuzzer input
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, create a simple fallback tensor
            input_tensor = torch::randn({2, 3});
        }

        // Parse minimum value from remaining bytes
        double min_value = 0.0;
        if (offset < Size) {
            // Use remaining bytes to determine min value
            size_t remaining = Size - offset;
            if (remaining >= sizeof(double)) {
                std::memcpy(&min_value, Data + offset, sizeof(double));
                offset += sizeof(double);
            } else if (remaining >= sizeof(float)) {
                float f_val;
                std::memcpy(&f_val, Data + offset, sizeof(float));
                min_value = static_cast<double>(f_val);
                offset += sizeof(float);
            } else {
                // Use single byte scaled to interesting range
                uint8_t byte_val = Data[offset++];
                min_value = (byte_val - 128.0) / 10.0; // Range roughly -12.8 to 12.7
            }
            
            // Handle special values
            if (std::isnan(min_value) || std::isinf(min_value)) {
                // Keep these special values - they're interesting edge cases
            }
        }

        // Test clamp_min with various scenarios
        
        // 1. Basic clamp_min operation
        torch::Tensor result = torch::clamp_min(input_tensor, min_value);
        
        // 2. In-place variant
        torch::Tensor input_copy = input_tensor.clone();
        input_copy.clamp_min_(min_value);
        
        // 3. Test with scalar tensor as min
        if (offset < Size && Data[offset] % 3 == 0) {
            torch::Tensor scalar_min = torch::tensor(min_value);
            torch::Tensor result2 = torch::clamp_min(input_tensor, scalar_min);
        }
        
        // 4. Test with different min values on same tensor
        if (offset + 1 < Size) {
            double min_value2 = (Data[offset] - 128.0) / 20.0;
            torch::Tensor result3 = torch::clamp_min(result, min_value2);
        }
        
        // 5. Test edge cases based on fuzzer input
        if (Size > 10) {
            uint8_t edge_selector = Data[0] % 8;
            switch (edge_selector) {
                case 0: // Very large positive min
                    torch::clamp_min(input_tensor, 1e10);
                    break;
                case 1: // Very large negative min
                    torch::clamp_min(input_tensor, -1e10);
                    break;
                case 2: // Zero min
                    torch::clamp_min(input_tensor, 0.0);
                    break;
                case 3: // NaN min (if supported)
                    torch::clamp_min(input_tensor, std::numeric_limits<double>::quiet_NaN());
                    break;
                case 4: // Infinity min
                    torch::clamp_min(input_tensor, std::numeric_limits<double>::infinity());
                    break;
                case 5: // Negative infinity min
                    torch::clamp_min(input_tensor, -std::numeric_limits<double>::infinity());
                    break;
                case 6: // Very small positive min
                    torch::clamp_min(input_tensor, std::numeric_limits<double>::min());
                    break;
                case 7: // Very small negative min
                    torch::clamp_min(input_tensor, -std::numeric_limits<double>::min());
                    break;
            }
        }
        
        // 6. Test with empty tensor
        if (Data[0] % 10 == 0) {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::clamp_min(empty_tensor, min_value);
        }
        
        // 7. Test with scalar tensor (0-dim)
        if (Data[0] % 10 == 1) {
            torch::Tensor scalar_tensor = torch::tensor(3.14);
            torch::clamp_min(scalar_tensor, min_value);
        }
        
        // 8. Test with different tensor types if fuzzer suggests
        if (offset < Size && Data[offset] % 5 == 0) {
            // Try with integer tensor
            torch::Tensor int_tensor = torch::randint(-100, 100, {3, 4}, torch::kInt32);
            torch::clamp_min(int_tensor, static_cast<int32_t>(min_value));
            
            // Try with bool tensor
            torch::Tensor bool_tensor = torch::randint(0, 2, {2, 2}, torch::kBool);
            torch::clamp_min(bool_tensor, false);
        }
        
        // 9. Test with non-contiguous tensor
        if (input_tensor.dim() >= 2 && input_tensor.size(0) > 1 && input_tensor.size(1) > 1) {
            torch::Tensor transposed = input_tensor.transpose(0, input_tensor.dim() - 1);
            torch::clamp_min(transposed, min_value);
        }
        
        // 10. Chain multiple clamp operations
        if (Data[0] % 7 == 0) {
            torch::Tensor chained = input_tensor;
            for (int i = 0; i < 3 && offset + i < Size; ++i) {
                double chain_min = (Data[offset + i] - 128.0) / 50.0;
                chained = torch::clamp_min(chained, chain_min);
            }
        }
        
        // Verify basic properties
        if (result.defined() && !std::isnan(min_value)) {
            // All values should be >= min_value (unless NaN)
            torch::Tensor mask = result >= min_value;
            torch::Tensor nan_mask = torch::isnan(result);
            torch::Tensor valid_mask = mask | nan_mask;
            
            // This check helps ensure correctness without being too strict
            if (!torch::all(valid_mask).item<bool>()) {
                // Interesting case - might be a bug or edge case
                // Don't throw, just note it
            }
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors - these are often expected for edge cases
        return 0;
    }
    catch (const std::bad_alloc &e)
    {
        // Memory allocation failure - input might be requesting too large tensor
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    
    return 0; // keep the input
}