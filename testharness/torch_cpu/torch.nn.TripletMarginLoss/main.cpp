#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        if (Size < 20) {
            return 0;
        }
        
        // Create anchor tensor first
        torch::Tensor anchor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure anchor is floating point
        if (!anchor.is_floating_point()) {
            anchor = anchor.to(torch::kFloat32);
        }
        
        // Get the shape of anchor to create matching positive and negative tensors
        auto shape = anchor.sizes().vec();
        
        // Create positive and negative tensors with the same shape as anchor
        torch::Tensor positive = torch::randn(shape, torch::kFloat32);
        torch::Tensor negative = torch::randn(shape, torch::kFloat32);
        
        // Use some fuzzer data to perturb the tensors
        if (offset + 2 <= Size) {
            float pos_scale = static_cast<float>(Data[offset]) / 255.0f * 2.0f;
            float neg_scale = static_cast<float>(Data[offset + 1]) / 255.0f * 2.0f;
            offset += 2;
            
            positive = positive * pos_scale;
            negative = negative * neg_scale;
        }
        
        // Parse margin parameter
        double margin = 1.0;
        if (offset + sizeof(float) <= Size) {
            float margin_f;
            std::memcpy(&margin_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize margin to avoid NaN/Inf
            if (std::isfinite(margin_f)) {
                margin = std::abs(margin_f);
                // Clamp to reasonable range
                margin = std::min(margin, 100.0);
            }
        }
        
        // Parse p parameter (norm degree)
        double p = 2.0;
        if (offset + sizeof(float) <= Size) {
            float p_f;
            std::memcpy(&p_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize p to avoid NaN/Inf, p must be > 0
            if (std::isfinite(p_f) && p_f > 0) {
                p = std::min(static_cast<double>(std::abs(p_f)), 10.0);
            }
        }
        
        // Parse swap parameter
        bool swap = false;
        if (offset < Size) {
            swap = static_cast<bool>(Data[offset] & 0x01);
            offset++;
        }
        
        // Parse reduction parameter - track with an int for later comparison
        int reduction_type = 1; // 0=none, 1=mean, 2=sum
        torch::nn::TripletMarginLossOptions::reduction_t reduction = torch::kMean;
        if (offset < Size) {
            reduction_type = Data[offset] % 3;
            offset++;
            
            switch (reduction_type) {
                case 0:
                    reduction = torch::kNone;
                    break;
                case 1:
                    reduction = torch::kMean;
                    break;
                case 2:
                    reduction = torch::kSum;
                    break;
            }
        }
        
        // Parse eps parameter
        double eps = 1e-6;
        if (offset < Size) {
            uint8_t eps_idx = Data[offset] % 4;
            offset++;
            switch (eps_idx) {
                case 0: eps = 1e-8; break;
                case 1: eps = 1e-6; break;
                case 2: eps = 1e-4; break;
                case 3: eps = 1e-2; break;
            }
        }
        
        // Create TripletMarginLoss module with the parsed parameters
        auto options = torch::nn::TripletMarginLossOptions()
            .margin(margin)
            .p(p)
            .eps(eps)
            .swap(swap)
            .reduction(reduction);
        
        auto triplet_loss = torch::nn::TripletMarginLoss(options);
        
        // Apply the loss function
        torch::Tensor loss;
        try {
            loss = triplet_loss->forward(anchor, positive, negative);
        } catch (const c10::Error& e) {
            // Shape mismatch or other expected errors - silently ignore
            return 0;
        }
        
        // Ensure the loss is computed by accessing its value
        if (reduction_type == 0) {
            // For no reduction, we get a tensor with the same batch size
            auto sum = loss.sum();
            volatile float loss_value = sum.item<float>();
            (void)loss_value;
        } else {
            volatile float loss_value = loss.item<float>();
            (void)loss_value;
        }
        
        // Also test with requires_grad to exercise backward path
        if (offset < Size && (Data[offset] & 0x01)) {
            torch::Tensor anchor_grad = anchor.clone().requires_grad_(true);
            torch::Tensor positive_grad = positive.clone().requires_grad_(true);
            torch::Tensor negative_grad = negative.clone().requires_grad_(true);
            
            try {
                auto loss_grad = triplet_loss->forward(anchor_grad, positive_grad, negative_grad);
                if (reduction_type != 0) {
                    loss_grad.backward();
                } else {
                    loss_grad.sum().backward();
                }
            } catch (const c10::Error& e) {
                // Gradient computation errors - silently ignore
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