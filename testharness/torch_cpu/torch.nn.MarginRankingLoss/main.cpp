#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isfinite

// --- Fuzzer Entry Point ---
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
        
        // Need at least some data to create tensors and parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensors for MarginRankingLoss
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input1 and input2 have the same shape
        // If shapes don't match, reshape input2 to match input1
        if (input1.sizes() != input2.sizes()) {
            try {
                input2 = input2.reshape(input1.sizes());
            } catch (...) {
                // If reshape fails, create a tensor with matching size
                input2 = torch::randn(input1.sizes());
            }
        }
        
        // Create target tensor with values -1 or 1
        // Target must have the same shape as inputs
        torch::Tensor target;
        if (offset < Size) {
            // Use fuzzer data to determine target values
            target = torch::empty(input1.sizes());
            auto target_accessor = target.flatten();
            int64_t num_elements = target_accessor.numel();
            for (int64_t i = 0; i < num_elements && (offset + i) < Size; i++) {
                // Map byte to -1 or 1
                float val = (Data[(offset + i) % Size] % 2 == 0) ? -1.0f : 1.0f;
                target_accessor[i] = val;
            }
            // Fill remaining elements if any
            for (int64_t i = Size - offset; i < num_elements; i++) {
                target_accessor[i] = (i % 2 == 0) ? -1.0f : 1.0f;
            }
            target = target_accessor.reshape(input1.sizes());
            offset += std::min(static_cast<size_t>(num_elements), Size - offset);
        } else {
            // Default target with alternating -1 and 1
            target = torch::ones(input1.sizes());
        }
        
        // Get margin value from the remaining data
        float margin = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&margin, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Clamp margin to reasonable range if not finite
            if (!std::isfinite(margin)) {
                margin = 0.0f;
            }
            // Clamp to reasonable range to avoid numerical issues
            margin = std::max(-100.0f, std::min(100.0f, margin));
        }
        
        // Get reduction mode from the remaining data
        uint8_t reduction_selector = 0;
        if (offset < Size) {
            reduction_selector = Data[offset++];
        }
        
        // Create MarginRankingLoss module with different reduction modes
        // Use torch::kNone, torch::kMean, torch::kSum for reduction_t type
        torch::Tensor loss;
        switch (reduction_selector % 3) {
            case 0: {
                auto options = torch::nn::MarginRankingLossOptions()
                                   .margin(margin)
                                   .reduction(torch::kNone);
                auto loss_fn = torch::nn::MarginRankingLoss(options);
                loss = loss_fn(input1, input2, target);
                break;
            }
            case 1: {
                auto options = torch::nn::MarginRankingLossOptions()
                                   .margin(margin)
                                   .reduction(torch::kMean);
                auto loss_fn = torch::nn::MarginRankingLoss(options);
                loss = loss_fn(input1, input2, target);
                break;
            }
            case 2: {
                auto options = torch::nn::MarginRankingLossOptions()
                                   .margin(margin)
                                   .reduction(torch::kSum);
                auto loss_fn = torch::nn::MarginRankingLoss(options);
                loss = loss_fn(input1, input2, target);
                break;
            }
            default: {
                auto options = torch::nn::MarginRankingLossOptions()
                                   .margin(margin)
                                   .reduction(torch::kMean);
                auto loss_fn = torch::nn::MarginRankingLoss(options);
                loss = loss_fn(input1, input2, target);
                break;
            }
        }
        
        // Ensure computation is performed
        if (loss.numel() == 1) {
            loss.item<float>();
        } else {
            // For reduction=None, sum the result to force computation
            loss.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}