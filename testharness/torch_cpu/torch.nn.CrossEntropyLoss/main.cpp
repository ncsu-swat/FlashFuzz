#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Parse dimensions for logits from fuzz data
        uint8_t batch_size_byte = Data[offset++];
        uint8_t num_classes_byte = Data[offset++];
        
        int64_t batch_size = (batch_size_byte % 16) + 1;  // 1-16
        int64_t num_classes = (num_classes_byte % 32) + 2; // 2-33 (need at least 2 classes)
        
        // Create logits tensor [batch_size, num_classes] with requires_grad
        torch::Tensor logits = torch::randn({batch_size, num_classes}, 
            torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
        
        // Vary logits based on fuzz data
        if (offset + 4 <= Size) {
            float scale;
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(scale) && std::abs(scale) < 100.0f) {
                logits = logits * scale;
            }
        }
        
        // Create target tensor (class indices in range [0, num_classes))
        torch::Tensor target = torch::randint(0, num_classes, {batch_size}, torch::kLong);
        
        // Modify target based on fuzz data
        if (offset + batch_size <= Size) {
            auto target_accessor = target.accessor<int64_t, 1>();
            for (int64_t i = 0; i < batch_size && offset < Size; i++) {
                target_accessor[i] = Data[offset++] % num_classes;
            }
        }
        
        // Parse whether to use weight
        torch::Tensor weight;
        bool use_weight = false;
        if (offset < Size) {
            use_weight = (Data[offset++] % 4 == 0); // 25% chance to use weight
            if (use_weight) {
                // Weight must have shape [num_classes]
                weight = torch::rand({num_classes}, torch::kFloat32) + 0.1f; // Avoid zero weights
            }
        }
        
        // Parse reduction method
        torch::Reduction::Reduction reduction = torch::Reduction::Mean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0:
                    reduction = torch::Reduction::None;
                    break;
                case 1:
                    reduction = torch::Reduction::Mean;
                    break;
                case 2:
                    reduction = torch::Reduction::Sum;
                    break;
            }
        }
        
        // Parse ignore_index
        int64_t ignore_index = -100; // Default value
        if (offset < Size) {
            uint8_t ignore_byte = Data[offset++];
            if (ignore_byte % 4 == 0) {
                // Sometimes use a valid class index to ignore
                ignore_index = ignore_byte % num_classes;
            }
            // Otherwise keep default -100
        }
        
        // Parse label_smoothing
        double label_smoothing = 0.0;
        if (offset < Size) {
            uint8_t smooth_byte = Data[offset++];
            label_smoothing = (smooth_byte % 101) / 100.0; // 0.0 to 1.0
        }
        
        // Build options
        auto options = torch::nn::CrossEntropyLossOptions()
            .ignore_index(ignore_index)
            .label_smoothing(label_smoothing);
        
        // Set reduction using the proper enum
        if (reduction == torch::Reduction::None) {
            options.reduction(torch::kNone);
        } else if (reduction == torch::Reduction::Sum) {
            options.reduction(torch::kSum);
        } else {
            options.reduction(torch::kMean);
        }
        
        if (use_weight && weight.defined()) {
            options.weight(weight);
        }
        
        // Create CrossEntropyLoss module
        torch::nn::CrossEntropyLoss criterion(options);
        
        // Apply the loss function
        torch::Tensor loss;
        try {
            loss = criterion(logits, target);
        } catch (const c10::Error&) {
            // Shape mismatch or other expected errors
            return 0;
        }
        
        // Optionally compute gradients
        if (offset < Size && Data[offset++] % 2 == 1) {
            try {
                if (reduction != torch::Reduction::None) {
                    // Scalar loss
                    loss.backward();
                } else {
                    // For reduction=none, loss has shape [batch_size]
                    torch::Tensor grad = torch::ones_like(loss);
                    loss.backward(grad);
                }
            } catch (const c10::Error&) {
                // Backward might fail in some cases
            }
        }
        
        // Test with probability targets (soft labels) - alternative input format
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                // Probability targets: shape [batch_size, num_classes]
                torch::Tensor prob_target = torch::softmax(
                    torch::randn({batch_size, num_classes}), /*dim=*/1);
                
                torch::Tensor logits2 = torch::randn({batch_size, num_classes},
                    torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
                
                torch::Tensor loss2 = criterion(logits2, prob_target);
            } catch (const c10::Error&) {
                // Soft labels may not be supported in all configurations
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