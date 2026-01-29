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
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor (predictions)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (ground truth)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            target = input.clone();
        }
        
        // Ensure input and target have the same shape
        if (input.sizes() != target.sizes()) {
            // Use expand or resize instead of reshape_as which may fail
            auto min_numel = std::min(input.numel(), target.numel());
            if (min_numel > 0) {
                input = input.flatten().slice(0, 0, min_numel);
                target = target.flatten().slice(0, 0, min_numel);
            } else {
                return 0;
            }
        }
        
        // Ensure tensors are float type for BCE
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        if (!target.is_floating_point()) {
            target = target.to(torch::kFloat32);
        }
        
        // Enable gradient tracking on input for backward pass
        input = input.clone().detach().requires_grad_(true);
        
        // Ensure input and target values are between 0 and 1 (required for BCE)
        input = torch::sigmoid(input);
        target = torch::sigmoid(target.detach());
        
        // Extract reduction mode from data if available
        torch::Reduction::Reduction reduction = torch::Reduction::Mean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0: reduction = torch::Reduction::None; break;
                case 1: reduction = torch::Reduction::Sum; break;
                case 2: reduction = torch::Reduction::Mean; break;
            }
        }
        
        // Extract weight tensor if there's data left
        torch::Tensor weight;
        bool use_weight = false;
        if (offset < Size) {
            uint8_t use_weight_byte = Data[offset++];
            use_weight = (use_weight_byte % 2 == 1);
            
            if (use_weight && offset < Size) {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
                if (!weight.is_floating_point()) {
                    weight = weight.to(torch::kFloat32);
                }
                // Match weight shape to input
                if (weight.numel() != input.numel()) {
                    auto target_numel = input.numel();
                    if (weight.numel() > target_numel) {
                        weight = weight.flatten().slice(0, 0, target_numel).reshape(input.sizes());
                    } else {
                        // Repeat weight to match size
                        weight = weight.flatten();
                        auto repeats = (target_numel / weight.numel()) + 1;
                        weight = weight.repeat({repeats}).slice(0, 0, target_numel).reshape(input.sizes());
                    }
                } else if (weight.sizes() != input.sizes()) {
                    weight = weight.reshape(input.sizes());
                }
                // Weight must be positive
                weight = torch::abs(weight) + 1e-6;
            }
        }
        
        // Create BCE Loss module with different configurations
        torch::nn::BCELoss bce_loss;
        
        // Configure the BCE Loss based on available data
        torch::nn::BCELossOptions options;
        if (reduction == torch::Reduction::None) {
            options = options.reduction(torch::kNone);
        } else if (reduction == torch::Reduction::Sum) {
            options = options.reduction(torch::kSum);
        } else {
            options = options.reduction(torch::kMean);
        }
        
        if (use_weight && weight.defined()) {
            options = options.weight(weight);
        }
        
        bce_loss = torch::nn::BCELoss(options);
        
        // Apply the BCE Loss
        torch::Tensor loss = bce_loss->forward(input, target);
        
        // Try backward pass
        try {
            if (loss.dim() == 0) {
                loss.backward();
            } else {
                loss.sum().backward();
            }
        } catch (...) {
            // Ignore backward pass errors silently
        }
        
        // Try with different reduction options if there's more data
        if (offset + 1 < Size) {
            uint8_t option_byte = Data[offset++];
            
            torch::Reduction::Reduction alt_reduction;
            switch (option_byte % 3) {
                case 0: alt_reduction = torch::Reduction::None; break;
                case 1: alt_reduction = torch::Reduction::Sum; break;
                case 2: alt_reduction = torch::Reduction::Mean; break;
            }
            
            if (alt_reduction != reduction) {
                try {
                    torch::nn::BCELossOptions alt_options;
                    if (alt_reduction == torch::Reduction::None) {
                        alt_options = alt_options.reduction(torch::kNone);
                    } else if (alt_reduction == torch::Reduction::Sum) {
                        alt_options = alt_options.reduction(torch::kSum);
                    } else {
                        alt_options = alt_options.reduction(torch::kMean);
                    }
                    torch::nn::BCELoss alt_bce_loss(alt_options);
                    torch::Tensor alt_loss = alt_bce_loss->forward(input.detach(), target);
                } catch (...) {
                    // Silently ignore alternative configuration errors
                }
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