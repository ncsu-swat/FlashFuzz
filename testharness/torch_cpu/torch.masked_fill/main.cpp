#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create mask tensor with same shape as input (must be boolean type)
        torch::Tensor mask = torch::zeros_like(input_tensor, torch::kBool);
        
        // Fill mask based on fuzz data
        if (offset < Size) {
            // Create a random pattern for the mask based on fuzz data
            auto mask_accessor = mask.flatten();
            int64_t mask_size = mask_accessor.numel();
            for (int64_t i = 0; i < mask_size && offset < Size; i++) {
                mask.flatten()[i] = static_cast<bool>(Data[offset++] & 1);
            }
        }
        
        // Get value to fill
        torch::Scalar value;
        if (offset + sizeof(float) <= Size) {
            float val;
            std::memcpy(&val, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize NaN/Inf to avoid unpredictable behavior
            if (std::isnan(val) || std::isinf(val)) {
                val = 0.0f;
            }
            value = torch::Scalar(val);
        } else {
            value = torch::Scalar(0.0f);
        }
        
        // Apply masked_fill operation (out-of-place)
        torch::Tensor result = input_tensor.masked_fill(mask, value);
        
        // Try masked_fill_ (in-place version)
        if (offset < Size) {
            torch::Tensor input_copy = input_tensor.clone();
            
            torch::Scalar value2;
            if (offset + sizeof(float) <= Size) {
                float val;
                std::memcpy(&val, Data + offset, sizeof(float));
                offset += sizeof(float);
                if (std::isnan(val) || std::isinf(val)) {
                    val = 1.0f;
                }
                value2 = torch::Scalar(val);
            } else {
                value2 = torch::Scalar(1.0f);
            }
            
            input_copy.masked_fill_(mask, value2);
        }
        
        // Try with different scalar types
        if (offset + 1 < Size) {
            uint8_t scalar_type = Data[offset++];
            
            torch::Scalar typed_value;
            switch (scalar_type % 5) {
                case 0:
                    typed_value = torch::Scalar(static_cast<int64_t>(Data[offset]));
                    break;
                case 1:
                    typed_value = torch::Scalar(static_cast<double>(Data[offset]));
                    break;
                case 2:
                    typed_value = torch::Scalar(static_cast<bool>(Data[offset] & 1));
                    break;
                case 3: {
                    int16_t val = 0;
                    if (offset + sizeof(int16_t) <= Size) {
                        std::memcpy(&val, Data + offset, sizeof(int16_t));
                    }
                    typed_value = torch::Scalar(val);
                    break;
                }
                case 4: {
                    int32_t val = 0;
                    if (offset + sizeof(int32_t) <= Size) {
                        std::memcpy(&val, Data + offset, sizeof(int32_t));
                    }
                    typed_value = torch::Scalar(val);
                    break;
                }
            }
            
            torch::Tensor result2 = input_tensor.masked_fill(mask, typed_value);
        }
        
        // Try with broadcastable mask shapes
        if (offset < Size && input_tensor.dim() > 0) {
            try {
                // Create a 1D mask that can broadcast to input shape
                int64_t last_dim = input_tensor.size(-1);
                torch::Tensor broadcast_mask = torch::zeros({last_dim}, torch::kBool);
                for (int64_t i = 0; i < last_dim && offset < Size; i++) {
                    broadcast_mask[i] = static_cast<bool>(Data[offset++] & 1);
                }
                torch::Tensor result3 = input_tensor.masked_fill(broadcast_mask, value);
            } catch (...) {
                // Shape incompatibility is expected in some cases
            }
        }
        
        // Try with a scalar mask (single boolean value)
        if (offset < Size) {
            bool scalar_mask_val = static_cast<bool>(Data[offset++] & 1);
            torch::Tensor scalar_mask_tensor = torch::tensor(scalar_mask_val);
            torch::Tensor result4 = input_tensor.masked_fill(scalar_mask_tensor, value);
        }
        
        // Try with all-true and all-false masks
        if (offset < Size) {
            uint8_t mask_choice = Data[offset++] % 2;
            torch::Tensor uniform_mask = mask_choice ? 
                torch::ones_like(input_tensor, torch::kBool) : 
                torch::zeros_like(input_tensor, torch::kBool);
            torch::Tensor result5 = input_tensor.masked_fill(uniform_mask, value);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}