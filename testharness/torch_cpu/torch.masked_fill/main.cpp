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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create mask tensor (must be boolean type)
        torch::Tensor mask;
        if (offset < Size) {
            // Try to create mask with same shape as input
            mask = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert to boolean type if not already
            mask = mask.to(torch::kBool);
        } else {
            // If we don't have enough data, create a simple mask
            mask = torch::ones_like(input_tensor, torch::kBool);
        }
        
        // Get value to fill
        torch::Scalar value;
        if (offset + sizeof(float) <= Size) {
            float val;
            std::memcpy(&val, Data + offset, sizeof(float));
            offset += sizeof(float);
            value = torch::Scalar(val);
        } else {
            value = torch::Scalar(0.0f);
        }
        
        // Apply masked_fill operation
        torch::Tensor result = input_tensor.masked_fill(mask, value);
        
        // Try masked_fill_ (in-place version)
        if (offset < Size) {
            // Create a copy of the input tensor for in-place operation
            torch::Tensor input_copy = input_tensor.clone();
            
            // Get another value for in-place operation
            torch::Scalar value2;
            if (offset + sizeof(float) <= Size) {
                float val;
                std::memcpy(&val, Data + offset, sizeof(float));
                offset += sizeof(float);
                value2 = torch::Scalar(val);
            } else {
                value2 = torch::Scalar(1.0f);
            }
            
            // Apply in-place masked_fill
            input_copy.masked_fill_(mask, value2);
        }
        
        // Try with different scalar types
        if (offset + 1 < Size) {
            uint8_t scalar_type = Data[offset++];
            
            // Create a scalar of different type based on the input data
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
            
            // Apply masked_fill with the typed scalar
            torch::Tensor result2 = input_tensor.masked_fill(mask, typed_value);
        }
        
        // Try with different mask shapes
        if (offset < Size) {
            // Create a mask with potentially different shape
            torch::Tensor alt_mask = fuzzer_utils::createTensor(Data, Size, offset);
            alt_mask = alt_mask.to(torch::kBool);
            
            // Try to apply masked_fill with this mask (may throw if shapes incompatible)
            try {
                torch::Tensor result3 = input_tensor.masked_fill(alt_mask, value);
            } catch (const c10::Error &e) {
                // Expected exception for incompatible shapes, just continue
            }
        }
        
        // Try with a scalar mask (single boolean value converted to tensor)
        if (offset < Size) {
            bool scalar_mask = static_cast<bool>(Data[offset] & 1);
            torch::Tensor scalar_mask_tensor = torch::tensor(scalar_mask);
            torch::Tensor result4 = input_tensor.masked_fill(scalar_mask_tensor, value);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
