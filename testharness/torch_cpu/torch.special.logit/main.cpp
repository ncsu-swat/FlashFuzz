#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.logit operation
        // logit(x) = log(x / (1 - x))
        // The operation expects input values in range (0, 1)
        
        // 1. Default version (no eps)
        torch::Tensor result1 = torch::special::logit(input);
        
        // 2. With eps parameter using at::logit (clamps values to [eps, 1-eps])
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure eps is positive and small
            float eps = std::abs(eps_raw);
            eps = std::fmod(eps, 0.5f);
            
            // Use at::logit which accepts the eps parameter
            torch::Tensor result2 = at::logit(input, static_cast<double>(eps));
        }
        
        // 3. Try with out parameter (no eps)
        torch::Tensor out = torch::empty_like(input);
        torch::special::logit_out(out, input);
        
        // 4. Try with different input types
        if (offset < Size) {
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
            torch::Tensor input_cast = input.to(dtype);
            
            try {
                torch::Tensor result3 = torch::special::logit(input_cast);
            } catch (const std::exception&) {
                // Some dtypes might not be supported, that's fine
            }
        }
        
        // 5. Try with non-contiguous tensor
        if (input.dim() > 1 && input.size(0) > 1 && input.size(1) > 1) {
            torch::Tensor transposed = input.transpose(0, 1);
            if (!transposed.is_contiguous()) {
                try {
                    torch::Tensor result4 = torch::special::logit(transposed);
                } catch (const std::exception&) {
                    // Non-contiguous input might fail in some cases
                }
            }
        }
        
        // 6. Test with values clamped to valid range (0, 1) explicitly
        if (offset < Size) {
            torch::Tensor clamped = torch::clamp(input, 0.01, 0.99);
            torch::Tensor result5 = torch::special::logit(clamped);
        }
        
        // 7. Test with eps and out parameter combined using at::logit_out
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            float eps = std::abs(eps_raw);
            eps = std::fmod(eps, 0.5f);
            
            torch::Tensor out2 = torch::empty_like(input);
            // Use at::logit_out which accepts the eps parameter
            at::logit_out(out2, input, static_cast<double>(eps));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}