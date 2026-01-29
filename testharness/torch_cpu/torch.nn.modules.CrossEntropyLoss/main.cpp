#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor (logits/predictions) with requires_grad for backward
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is floating point for cross entropy
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // CrossEntropyLoss expects at least 2D input: [batch, classes, ...]
        if (input.dim() < 2) {
            // Reshape to [1, N] where N is the number of elements
            int64_t n = input.numel();
            if (n == 0) n = 1;
            input = input.reshape({1, n});
        }
        
        // Enable gradient computation
        input = input.clone().detach().requires_grad_(true);
        
        // Create target tensor (ground truth labels)
        std::vector<int64_t> target_shape;
        target_shape.push_back(input.size(0)); // batch dimension
        for (int64_t i = 2; i < input.dim(); i++) {
            target_shape.push_back(input.size(i));
        }
        
        int64_t num_classes = input.size(1);
        if (num_classes <= 0) {
            num_classes = 1;
        }
        
        // Create target with valid class indices
        torch::Tensor target;
        if (offset < Size && Size - offset >= 4) {
            // Use fuzzer data to create target values
            torch::Tensor temp = fuzzer_utils::createTensor(Data, Size, offset);
            target = temp.flatten().to(torch::kLong);
            
            // Resize to match expected shape
            int64_t target_numel = 1;
            for (auto s : target_shape) target_numel *= s;
            
            if (target.numel() < target_numel) {
                // Pad with zeros
                torch::Tensor padded = torch::zeros({target_numel}, torch::kLong);
                padded.slice(0, 0, target.numel()) = target.slice(0, 0, std::min(target.numel(), target_numel));
                target = padded;
            }
            target = target.slice(0, 0, target_numel).reshape(target_shape);
            
            // Clamp to valid class indices
            target = torch::clamp(torch::abs(target), 0, num_classes - 1);
        } else {
            // Create random valid target
            target = torch::randint(0, num_classes, target_shape);
        }
        
        // Parse options from remaining data
        bool use_weight = false;
        float weight_value = 1.0f;
        int64_t ignore_index = -100;
        float smoothing_value = 0.0;
        uint8_t reduction_mode = 1;
        
        if (offset + 5 <= Size) {
            use_weight = Data[offset++] & 0x1;
            weight_value = static_cast<float>(Data[offset++]) / 255.0f * 10.0f + 0.01f;
            ignore_index = static_cast<int64_t>(Data[offset++]) - 100;
            smoothing_value = static_cast<float>(Data[offset++] % 100) / 100.0f;
            reduction_mode = Data[offset++] % 3;
        }
        
        // Create weight tensor if needed
        torch::Tensor weight;
        if (use_weight && num_classes > 0) {
            weight = torch::ones({num_classes}) * weight_value;
        }
        
        // Create CrossEntropyLoss options with appropriate reduction
        auto options = torch::nn::CrossEntropyLossOptions()
            .ignore_index(ignore_index)
            .label_smoothing(smoothing_value);
        
        // Set reduction mode
        switch (reduction_mode) {
            case 0: options.reduction(torch::kNone); break;
            case 1: options.reduction(torch::kMean); break;
            case 2: options.reduction(torch::kSum); break;
            default: options.reduction(torch::kMean); break;
        }
        
        if (use_weight && weight.defined()) {
            options.weight(weight);
        }
        
        // Create CrossEntropyLoss module
        torch::nn::CrossEntropyLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor loss = loss_fn->forward(input, target);
        
        // Compute gradients if possible (inner try-catch for expected failures)
        try {
            if (loss.defined() && loss.numel() > 0) {
                torch::Tensor loss_for_backward;
                if (reduction_mode == 0) {  // kNone
                    loss_for_backward = loss.sum();
                } else {
                    loss_for_backward = loss;
                }
                
                if (!loss_for_backward.isnan().any().item<bool>() && 
                    !loss_for_backward.isinf().any().item<bool>()) {
                    loss_for_backward.backward();
                }
            }
        } catch (...) {
            // Gradient computation may fail for various reasons, that's okay
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}