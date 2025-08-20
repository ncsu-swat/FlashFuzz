#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor (logits/predictions)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (ground truth labels)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, create a compatible target
            if (input.dim() >= 2) {
                // For typical case: [batch_size, num_classes, ...] -> [batch_size, ...]
                std::vector<int64_t> target_shape;
                target_shape.push_back(input.size(0)); // batch dimension
                for (int64_t i = 2; i < input.dim(); i++) {
                    target_shape.push_back(input.size(i));
                }
                
                // Create target with long dtype (for class indices)
                target = torch::zeros(target_shape, torch::kLong);
                
                // Fill with random class indices
                int64_t num_classes = input.size(1);
                if (num_classes > 0) {
                    target = torch::randint(0, num_classes, target_shape);
                }
            } else {
                // For unusual input shapes, create a simple target
                target = torch::zeros({1}, torch::kLong);
            }
        }
        
        // Try to make target compatible with input if possible
        if (input.dim() >= 2 && target.dim() > 0) {
            // Ensure target has proper dtype for class indices
            target = target.to(torch::kLong);
            
            // Clamp target values to valid class indices if needed
            if (input.size(1) > 0) {
                target = torch::clamp(target, 0, input.size(1) - 1);
            }
        }
        
        // Parse options from remaining data
        bool weight_reduction = false;
        float weight_value = 1.0f;
        int64_t ignore_index = -100;
        bool label_smoothing = false;
        float smoothing_value = 0.0;
        
        if (offset + 4 <= Size) {
            weight_reduction = Data[offset++] & 0x1;
            weight_value = static_cast<float>(Data[offset++]) / 255.0f * 10.0f;
            ignore_index = static_cast<int64_t>(Data[offset++]) - 100;
            label_smoothing = Data[offset++] & 0x1;
            smoothing_value = static_cast<float>(Data[offset] % 100) / 100.0f;
        }
        
        // Create weight tensor if needed
        torch::Tensor weight;
        if (weight_reduction && input.dim() >= 2) {
            int64_t num_classes = input.size(1);
            weight = torch::ones({num_classes}) * weight_value;
        }
        
        // Parse reduction mode
        torch::enumtype::Reduction reduction = torch::kMean;
        if (offset < Size) {
            uint8_t red_byte = Data[offset++];
            switch (red_byte % 3) {
                case 0: reduction = torch::kNone; break;
                case 1: reduction = torch::kMean; break;
                case 2: reduction = torch::kSum; break;
            }
        }
        
        // Create CrossEntropyLoss with various configurations
        torch::nn::CrossEntropyLoss loss_fn;
        
        // Configure with options
        if (weight_reduction && weight.defined()) {
            if (label_smoothing) {
                loss_fn = torch::nn::CrossEntropyLoss(
                    torch::nn::CrossEntropyLossOptions()
                        .weight(weight)
                        .ignore_index(ignore_index)
                        .reduction(reduction)
                        .label_smoothing(smoothing_value)
                );
            } else {
                loss_fn = torch::nn::CrossEntropyLoss(
                    torch::nn::CrossEntropyLossOptions()
                        .weight(weight)
                        .ignore_index(ignore_index)
                        .reduction(reduction)
                );
            }
        } else if (label_smoothing) {
            loss_fn = torch::nn::CrossEntropyLoss(
                torch::nn::CrossEntropyLossOptions()
                    .ignore_index(ignore_index)
                    .reduction(reduction)
                    .label_smoothing(smoothing_value)
            );
        } else {
            loss_fn = torch::nn::CrossEntropyLoss(
                torch::nn::CrossEntropyLossOptions()
                    .ignore_index(ignore_index)
                    .reduction(reduction)
            );
        }
        
        // Apply the loss function
        torch::Tensor loss = loss_fn->forward(input, target);
        
        // Compute gradients if possible
        if (loss.numel() > 0 && !loss.isnan().any().item<bool>() && 
            !loss.isinf().any().item<bool>()) {
            if (reduction != torch::kNone) {
                loss.backward();
            } else if (loss.dim() > 0 && loss.numel() > 0) {
                // For "none" reduction, we need to sum before backward
                loss.sum().backward();
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}