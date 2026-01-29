#include "fuzzer_utils.h"
#include <iostream>

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
        
        torch::Tensor weights = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure weights are non-negative and convert to float (multinomial requires float)
        weights = torch::abs(weights).to(torch::kFloat32);
        
        int64_t num_samples = 1;
        bool replacement = false;
        
        if (offset + 2 <= Size) {
            num_samples = static_cast<int64_t>(Data[offset++]) % 10 + 1;
            replacement = Data[offset++] & 0x1;
        }
        
        // Reshape weights to be 1D or 2D (multinomial requirement)
        if (weights.dim() == 0) {
            weights = weights.reshape({1});
        } else if (weights.dim() > 2) {
            if (offset < Size && Data[offset++] % 2 == 0) {
                weights = weights.flatten();
            } else {
                int64_t last_dim = weights.size(-1);
                weights = weights.reshape({-1, last_dim});
            }
        }
        
        // Ensure at least one positive weight to avoid "invalid multinomial distribution" error
        // Add a small epsilon to ensure sum > 0
        weights = weights + 1e-6f;
        
        // Get the number of categories (last dimension size)
        int64_t num_categories = weights.size(-1);
        
        // If replacement is false, num_samples cannot exceed num_categories
        if (!replacement && num_samples > num_categories) {
            num_samples = num_categories;
        }
        
        // Ensure num_samples is at least 1 and num_categories is at least 1
        if (num_samples < 1) {
            num_samples = 1;
        }
        if (num_categories < 1) {
            return 0;
        }
        
        torch::Tensor result;
        
        // Inner try-catch for expected failures (don't log these)
        try {
            if (offset < Size) {
                uint8_t variant = Data[offset++] % 2;
                
                switch (variant) {
                    case 0:
                        result = torch::multinomial(weights, num_samples, replacement);
                        break;
                        
                    case 1:
                        {
                            auto gen = torch::make_generator<torch::CPUGeneratorImpl>();
                            if (offset < Size) {
                                gen.set_current_seed(static_cast<uint64_t>(Data[offset++]));
                            }
                            result = torch::multinomial(weights, num_samples, replacement, gen);
                        }
                        break;
                }
            } else {
                result = torch::multinomial(weights, num_samples, replacement);
            }
            
            // Access result to ensure computation is not optimized away
            if (result.numel() > 0) {
                auto sum = result.sum().item<int64_t>();
                (void)sum;
            }
        }
        catch (const c10::Error &e) {
            // Expected errors from invalid tensor configurations - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}