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
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor (log probabilities)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has 4 dimensions (N, C, H, W) for 2D NLLLoss
        while (input.dim() < 4) {
            input = input.unsqueeze(0);
        }
        // If too many dimensions, reshape to 4D
        if (input.dim() > 4) {
            auto sizes = input.sizes().vec();
            int64_t total = input.numel();
            // Reshape to [1, C, H, W] where we try to preserve some structure
            int64_t c = std::max<int64_t>(1, sizes[0]);
            int64_t remaining = total / c;
            int64_t h = std::max<int64_t>(1, static_cast<int64_t>(std::sqrt(remaining)));
            int64_t w = std::max<int64_t>(1, remaining / h);
            input = input.flatten().slice(0, 0, c * h * w).reshape({1, c, h, w});
        }
        
        // Ensure we have at least 1 class
        if (input.size(1) < 1) {
            return 0;
        }
        
        // Convert input to float and apply log_softmax to get proper log probabilities
        input = input.to(torch::kFloat);
        input = torch::log_softmax(input, /*dim=*/1);
        
        int64_t N = input.size(0);
        int64_t C = input.size(1);
        int64_t H = input.size(2);
        int64_t W = input.size(3);
        
        // Create target tensor with shape [N, H, W] and valid class indices [0, C-1]
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            target = target.flatten();
            int64_t needed = N * H * W;
            if (target.numel() < needed) {
                target = target.repeat((needed / target.numel()) + 1);
            }
            target = target.slice(0, 0, needed).reshape({N, H, W});
        } else {
            target = torch::zeros({N, H, W});
        }
        
        // Convert target to long and clamp to valid range [0, C-1]
        target = target.to(torch::kFloat).abs();
        target = target.fmod(static_cast<float>(C)).to(torch::kLong);
        
        // Get weight tensor (optional)
        torch::Tensor weight;
        bool use_weight = false;
        if (offset < Size && Data[offset++] % 2 == 0) {
            use_weight = true;
            if (offset < Size) {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
                weight = weight.to(torch::kFloat).flatten();
                // Ensure weight has size C
                if (weight.numel() < C) {
                    weight = weight.repeat((C / weight.numel()) + 1);
                }
                weight = weight.slice(0, 0, C).abs() + 0.01f; // Ensure positive weights
            } else {
                weight = torch::ones({C});
            }
        }
        
        // Parse reduction mode
        torch::nn::NLLLossOptions::reduction_t reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0:
                    reduction_mode = torch::kNone;
                    break;
                case 1:
                    reduction_mode = torch::kSum;
                    break;
                case 2:
                default:
                    reduction_mode = torch::kMean;
                    break;
            }
        }
        
        // Parse ignore_index - limit to reasonable range
        int64_t ignore_index = -100;
        if (offset < Size) {
            int8_t idx_byte;
            std::memcpy(&idx_byte, Data + offset, 1);
            offset++;
            // Map to range [-100, C] to include valid ignore scenarios
            ignore_index = (idx_byte % (C + 101)) - 100;
        }
        
        // Create NLLLoss options (NLLLoss2d is just NLLLoss in PyTorch)
        auto options = torch::nn::NLLLossOptions()
            .reduction(reduction_mode)
            .ignore_index(ignore_index);
        
        if (use_weight) {
            options = options.weight(weight);
        }
        
        // Create NLLLoss module (NLLLoss2d is deprecated alias for NLLLoss)
        torch::nn::NLLLoss nll_loss(options);
        
        // Apply NLLLoss - this works for 2D spatial inputs
        try {
            torch::Tensor output = nll_loss(input, target);
            
            // Ensure we use the output to prevent optimization
            if (output.defined()) {
                volatile float sum = output.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error&) {
            // Shape mismatch or other torch errors - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}