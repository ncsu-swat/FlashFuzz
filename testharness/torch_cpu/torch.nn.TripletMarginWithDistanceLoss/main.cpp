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
        // Need sufficient data for options and tensors
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse options first
        double margin = static_cast<double>(Data[offset]) / 25.0 + 0.1; // margin > 0
        offset++;
        
        bool swap = Data[offset] % 2 == 1;
        offset++;
        
        uint8_t red_val = Data[offset] % 3;
        offset++;
        
        // Determine batch size and embedding dimension from fuzz data
        int64_t batch_size = (Data[offset] % 8) + 1;  // 1-8
        offset++;
        
        int64_t embed_dim = (Data[offset] % 16) + 1;  // 1-16
        offset++;
        
        // Create anchor, positive, and negative tensors with same shape
        // TripletMarginWithDistanceLoss expects tensors of shape (N, D) or (D,)
        torch::Tensor anchor = torch::randn({batch_size, embed_dim}, torch::kFloat32);
        torch::Tensor positive = torch::randn({batch_size, embed_dim}, torch::kFloat32);
        torch::Tensor negative = torch::randn({batch_size, embed_dim}, torch::kFloat32);
        
        // Use remaining fuzz data to perturb tensors
        if (offset + 3 <= Size) {
            float scale = static_cast<float>(Data[offset]) / 255.0f * 2.0f;
            anchor = anchor * scale;
            offset++;
            
            scale = static_cast<float>(Data[offset]) / 255.0f * 2.0f;
            positive = positive * scale;
            offset++;
            
            scale = static_cast<float>(Data[offset]) / 255.0f * 2.0f;
            negative = negative * scale;
            offset++;
        }
        
        // Build options
        auto options = torch::nn::TripletMarginWithDistanceLossOptions()
            .margin(margin)
            .swap(swap);
        
        // Set reduction
        if (red_val == 0) {
            options.reduction(torch::kNone);
        } else if (red_val == 1) {
            options.reduction(torch::kMean);
        } else {
            options.reduction(torch::kSum);
        }
        
        // Create the TripletMarginWithDistanceLoss module
        torch::nn::TripletMarginWithDistanceLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor loss = loss_fn(anchor, positive, negative);
        
        // Ensure computation is performed
        // Sum handles both scalar and non-scalar outputs
        float loss_val = loss.sum().item<float>();
        (void)loss_val;
        
        // Test backward pass
        if (offset < Size && Data[offset] % 2 == 0) {
            anchor = anchor.detach().requires_grad_(true);
            positive = positive.detach().requires_grad_(true);
            negative = negative.detach().requires_grad_(true);
            
            // Recompute loss with grad-enabled tensors
            torch::Tensor loss_grad = loss_fn(anchor, positive, negative);
            
            // Need scalar for backward
            torch::Tensor loss_scalar = loss_grad.sum();
            loss_scalar.backward();
            
            // Access gradients to ensure computation
            auto grad_a = anchor.grad();
            auto grad_p = positive.grad();
            auto grad_n = negative.grad();
            (void)grad_a;
            (void)grad_p;
            (void)grad_n;
        }
        
        // Test with custom distance function if more data available
        if (offset + 1 < Size && Data[offset] % 3 == 0) {
            // Create module with custom distance function (pairwise distance)
            auto custom_options = torch::nn::TripletMarginWithDistanceLossOptions()
                .distance_function([](const torch::Tensor& x, const torch::Tensor& y) {
                    return torch::pairwise_distance(x, y);
                })
                .margin(margin)
                .swap(swap);
            
            if (red_val == 0) {
                custom_options.reduction(torch::kNone);
            } else if (red_val == 1) {
                custom_options.reduction(torch::kMean);
            } else {
                custom_options.reduction(torch::kSum);
            }
            
            torch::nn::TripletMarginWithDistanceLoss custom_loss_fn(custom_options);
            
            try {
                torch::Tensor custom_loss = custom_loss_fn(anchor.detach(), positive.detach(), negative.detach());
                float custom_loss_val = custom_loss.sum().item<float>();
                (void)custom_loss_val;
            } catch (...) {
                // Custom distance function may fail for certain inputs
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