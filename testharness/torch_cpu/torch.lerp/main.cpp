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
        
        if (Size < 4) {
            return 0;
        }
        
        // Read control bytes upfront
        uint8_t use_scalar_weight_byte = Data[offset++];
        uint8_t variant_byte = Data[offset++];
        
        // Create input tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create end tensor with same shape as input for broadcasting compatibility
        torch::Tensor end;
        if (offset < Size) {
            end = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            end = torch::ones_like(input);
        }
        
        // Create weight tensor or scalar
        torch::Tensor weight;
        float weight_scalar = 0.5f;
        bool use_scalar_weight = (use_scalar_weight_byte % 2 == 0);
        
        if (use_scalar_weight) {
            // Use next 4 bytes as float if available
            if (offset + sizeof(float) <= Size) {
                memcpy(&weight_scalar, Data + offset, sizeof(float));
                offset += sizeof(float);
                // Sanitize the weight to avoid NaN/Inf issues
                if (!std::isfinite(weight_scalar)) {
                    weight_scalar = 0.5f;
                }
            }
        } else {
            // Create weight tensor
            if (offset < Size) {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                weight = torch::rand_like(input);
            }
        }
        
        // Apply torch.lerp operation
        torch::Tensor result;
        int variant = variant_byte % 3;
        
        try {
            if (variant == 0) {
                // Variant 1: input.lerp(end, weight)
                if (use_scalar_weight) {
                    result = input.lerp(end, weight_scalar);
                } else {
                    result = input.lerp(end, weight);
                }
            } else if (variant == 1) {
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
        catch (const c10::Error &e) {
            // Expected errors from shape mismatches, dtype issues, etc.
            // Silently ignore these
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}