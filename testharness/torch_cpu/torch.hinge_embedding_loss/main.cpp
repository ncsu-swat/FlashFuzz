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
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor with same shape as input
        // Target must contain only 1 or -1 values
        torch::Tensor target_raw = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape target to match input shape and convert to {-1, 1} values
        try {
            target_raw = target_raw.reshape(input.sizes());
        } catch (...) {
            // If reshape fails, create a new tensor with matching shape
            target_raw = torch::randn(input.sizes());
        }
        
        // Convert target to only contain -1 or 1 values (as required by hinge_embedding_loss)
        torch::Tensor target = torch::where(target_raw >= 0, 
                                            torch::ones_like(target_raw), 
                                            -torch::ones_like(target_raw));
        
        // Extract margin parameter from the input data
        double margin = 1.0;
        if (offset + sizeof(float) <= Size) {
            float margin_val;
            std::memcpy(&margin_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize margin to avoid extreme values
            if (std::isfinite(margin_val)) {
                margin = static_cast<double>(margin_val);
                // Clamp margin to reasonable range
                margin = std::max(-100.0, std::min(100.0, margin));
            }
        }
        
        // Extract reduction parameter from the input data
        int64_t reduction_raw = 1; // Default to mean
        if (offset + sizeof(int8_t) <= Size) {
            int8_t reduction_val;
            std::memcpy(&reduction_val, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            reduction_raw = std::abs(static_cast<int64_t>(reduction_val)) % 3;
        }
        
        // Map reduction value to at::Reduction enum
        int64_t reduction;
        switch (reduction_raw) {
            case 0:
                reduction = at::Reduction::None;
                break;
            case 1:
                reduction = at::Reduction::Mean;
                break;
            case 2:
                reduction = at::Reduction::Sum;
                break;
            default:
                reduction = at::Reduction::Mean;
        }
        
        // Apply hinge_embedding_loss
        torch::Tensor result = torch::hinge_embedding_loss(input, target, margin, reduction);
        
        // Verify result is valid by computing sum (works for both scalar and tensor results)
        volatile float check = result.sum().item<float>();
        (void)check;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}