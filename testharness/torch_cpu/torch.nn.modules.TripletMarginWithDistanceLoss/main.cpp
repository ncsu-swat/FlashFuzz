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
        // Need at least some data to proceed
        if (Size < 20) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse options first
        double margin = 1.0;
        bool swap = false;
        torch::nn::TripletMarginWithDistanceLossOptions::reduction_t reduction = torch::kMean;
        
        if (offset + 3 <= Size) {
            uint8_t margin_byte = Data[offset++];
            margin = static_cast<double>(margin_byte) / 10.0;
            
            swap = Data[offset++] % 2 == 1;
            
            uint8_t reduction_byte = Data[offset++] % 3;
            switch (reduction_byte) {
                case 0: reduction = torch::kNone; break;
                case 1: reduction = torch::kMean; break;
                case 2: reduction = torch::kSum; break;
                default: reduction = torch::kMean;
            }
        }
        
        // Determine if we should test backward pass
        bool test_backward = false;
        if (offset < Size) {
            test_backward = (Data[offset++] % 2 == 0);
        }
        
        // Create anchor tensor first
        torch::Tensor anchor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (anchor.numel() == 0) {
            return 0;
        }
        
        // Create positive and negative tensors with the same shape as anchor
        auto sizes = anchor.sizes().vec();
        auto dtype = anchor.dtype();
        
        // Use remaining data to seed tensor values
        torch::Tensor positive, negative;
        
        try {
            // Create tensors with matching shapes
            positive = torch::randn(sizes, torch::TensorOptions().dtype(dtype));
            negative = torch::randn(sizes, torch::TensorOptions().dtype(dtype));
            
            // If we have data left, use it to modify tensors slightly
            if (offset < Size) {
                float scale = static_cast<float>(Data[offset++]) / 255.0f;
                positive = positive * scale;
            }
            if (offset < Size) {
                float scale = static_cast<float>(Data[offset++]) / 255.0f;
                negative = negative * scale;
            }
        } catch (...) {
            // Shape creation failed silently, use anchor clones
            positive = anchor.clone();
            negative = anchor.clone() + 0.1;
        }
        
        // Ensure tensors are float type for loss computation
        if (!anchor.is_floating_point()) {
            anchor = anchor.to(torch::kFloat32);
            positive = positive.to(torch::kFloat32);
            negative = negative.to(torch::kFloat32);
        }
        
        // Create the TripletMarginWithDistanceLoss module
        auto options = torch::nn::TripletMarginWithDistanceLossOptions()
            .margin(margin)
            .swap(swap)
            .reduction(reduction);
        
        torch::nn::TripletMarginWithDistanceLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor loss;
        {
            torch::NoGradGuard no_grad;
            loss = loss_fn(anchor, positive, negative);
        }
        
        // Ensure computation is performed
        if (loss.numel() == 1) {
            loss.item<float>();
        } else {
            loss.sum().item<float>();
        }
        
        // Test backward pass if requested
        if (test_backward) {
            try {
                auto anchor_grad = anchor.detach().clone().requires_grad_(true);
                auto positive_grad = positive.detach().clone().requires_grad_(true);
                auto negative_grad = negative.detach().clone().requires_grad_(true);
                
                // Use mean reduction for backward to ensure scalar output
                auto backward_options = torch::nn::TripletMarginWithDistanceLossOptions()
                    .margin(margin)
                    .swap(swap)
                    .reduction(torch::kMean);
                
                torch::nn::TripletMarginWithDistanceLoss loss_fn_backward(backward_options);
                auto loss_grad = loss_fn_backward(anchor_grad, positive_grad, negative_grad);
                loss_grad.backward();
            } catch (...) {
                // Backward may fail for various reasons, ignore silently
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