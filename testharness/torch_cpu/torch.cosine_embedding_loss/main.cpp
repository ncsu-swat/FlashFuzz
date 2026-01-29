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
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create first input tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left
        if (offset >= Size) {
            return 0;
        }
        
        // Create second input tensor
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input1 and input2 have the same shape (required by cosine_embedding_loss)
        // If shapes don't match, reshape input2 to match input1
        if (input1.sizes() != input2.sizes()) {
            try {
                input2 = input2.reshape(input1.sizes());
            } catch (...) {
                // If reshape fails, create input2 with same shape as input1
                input2 = torch::randn(input1.sizes());
            }
        }
        
        // Both inputs must be at least 2D for cosine_embedding_loss
        // The function computes cosine similarity along the last dimension
        if (input1.dim() < 2) {
            input1 = input1.unsqueeze(0);
            input2 = input2.unsqueeze(0);
        }
        
        // Get batch size (first dimension)
        int64_t batch_size = input1.size(0);
        
        // Create target tensor - must be 1D with size equal to batch_size
        // Values must be -1 or 1
        torch::Tensor target;
        if (offset < Size) {
            // Use remaining data to determine target values
            target = torch::empty({batch_size}, torch::kFloat32);
            for (int64_t i = 0; i < batch_size; i++) {
                if (offset < Size) {
                    // Use data byte to determine -1 or 1
                    target[i] = (Data[offset++] % 2 == 0) ? 1.0f : -1.0f;
                } else {
                    target[i] = 1.0f;
                }
            }
        } else {
            // Default target of all 1s
            target = torch::ones({batch_size});
        }
        
        // Get margin parameter from the remaining data
        float margin = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&margin, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Sanitize margin - handle NaN and Inf, clamp to reasonable range
            if (std::isnan(margin) || std::isinf(margin)) {
                margin = 0.0f;
            }
            // Clamp margin to typical range [-1, 1]
            margin = std::max(-1.0f, std::min(1.0f, margin));
        }
        
        // Get reduction mode from the remaining data
        int64_t reduction_mode = 1; // Default to mean
        if (offset < Size) {
            reduction_mode = static_cast<int64_t>(Data[offset++]) % 3;
        }
        
        // Map reduction_mode to at::Reduction
        int64_t reduction;
        switch (reduction_mode) {
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
        
        // Apply cosine_embedding_loss
        torch::Tensor loss = torch::cosine_embedding_loss(
            input1, 
            input2, 
            target, 
            margin, 
            reduction
        );
        
        // Ensure the loss is computed by accessing its value
        if (loss.numel() > 0) {
            if (loss.dim() == 0) {
                volatile float loss_value = loss.item<float>();
                (void)loss_value;
            } else {
                // For reduction=None, loss is a tensor
                volatile float sum_value = loss.sum().item<float>();
                (void)sum_value;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}