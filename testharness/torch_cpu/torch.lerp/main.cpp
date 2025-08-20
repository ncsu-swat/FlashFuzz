#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create end tensor with same shape as input
        torch::Tensor end;
        if (offset < Size) {
            end = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            end = torch::ones_like(input);
        }
        
        // Create weight tensor or scalar
        torch::Tensor weight;
        float weight_scalar = 0.5f;
        bool use_scalar_weight = false;
        
        if (offset < Size) {
            // Use the next byte to decide whether to use scalar or tensor weight
            use_scalar_weight = (Data[offset++] % 2 == 0);
            
            if (use_scalar_weight && offset < Size) {
                // Use next 4 bytes as float if available
                if (offset + sizeof(float) <= Size) {
                    memcpy(&weight_scalar, Data + offset, sizeof(float));
                    offset += sizeof(float);
                }
            } else {
                // Create weight tensor
                if (offset < Size) {
                    weight = fuzzer_utils::createTensor(Data, Size, offset);
                } else {
                    weight = torch::rand_like(input);
                }
            }
        } else {
            use_scalar_weight = true;
        }
        
        // Apply torch.lerp operation
        torch::Tensor result;
        
        // Try different variants of lerp
        if (offset < Size && Data[offset++] % 3 == 0) {
            // Variant 1: input.lerp(end, weight)
            if (use_scalar_weight) {
                result = input.lerp(end, weight_scalar);
            } else {
                result = input.lerp(end, weight);
            }
        } else if (offset < Size && Data[offset++] % 2 == 0) {
            // Variant 2: torch::lerp(input, end, weight)
            if (use_scalar_weight) {
                result = torch::lerp(input, end, weight_scalar);
            } else {
                result = torch::lerp(input, end, weight);
            }
        } else {
            // Variant 3: torch::lerp_out(result, input, end, weight)
            result = torch::empty_like(input);
            if (use_scalar_weight) {
                torch::lerp_out(result, input, end, weight_scalar);
            } else {
                torch::lerp_out(result, input, end, weight);
            }
        }
        
        // Force computation to ensure any potential errors are triggered
        result.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}