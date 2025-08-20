#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
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
            // If we don't have enough data for a second tensor, clone the first one
            target = input.clone();
        }
        
        // Try to make input and target have the same shape
        if (input.sizes() != target.sizes() && input.numel() > 0 && target.numel() > 0) {
            if (input.dim() > 0 && target.dim() > 0) {
                // Reshape target to match input if possible
                try {
                    target = target.reshape_as(input);
                } catch (...) {
                    // If reshape fails, try to create a new target with the same shape
                    target = torch::rand_like(input);
                }
            }
        }
        
        // Ensure input values are between 0 and 1 for BCE
        if (input.is_floating_point()) {
            input = torch::sigmoid(input);
        } else {
            // Convert to float and apply sigmoid
            input = torch::sigmoid(input.to(torch::kFloat));
        }
        
        // Ensure target values are between 0 and 1 for BCE
        if (target.is_floating_point()) {
            target = torch::clamp(target, 0.0, 1.0);
        } else {
            // Convert to float and clamp
            target = torch::clamp(target.to(torch::kFloat), 0.0, 1.0);
        }
        
        // Parse reduction mode from input data
        torch::nn::BCELossOptions::reduction_t reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0:
                    reduction_mode = torch::kMean;
                    break;
                case 1:
                    reduction_mode = torch::kSum;
                    break;
                case 2:
                    reduction_mode = torch::kNone;
                    break;
            }
        }
        
        // Parse weight tensor option
        bool use_weight = false;
        torch::Tensor weight;
        if (offset < Size) {
            use_weight = (Data[offset++] % 2 == 1);
            if (use_weight && offset < Size) {
                try {
                    weight = fuzzer_utils::createTensor(Data, Size, offset);
                    if (weight.sizes() != input.sizes() && weight.numel() > 0) {
                        try {
                            weight = weight.reshape_as(input);
                        } catch (...) {
                            weight = torch::ones_like(input);
                        }
                    }
                    if (!weight.is_floating_point()) {
                        weight = weight.to(torch::kFloat);
                    }
                    // Ensure weights are positive
                    weight = torch::abs(weight) + 0.1;
                } catch (...) {
                    use_weight = false;
                }
            }
        }
        
        // Create BCE Loss module with different configurations
        torch::nn::BCELoss bce_loss;
        if (use_weight) {
            bce_loss = torch::nn::BCELoss(torch::nn::BCELossOptions().weight(weight).reduction(reduction_mode));
        } else {
            bce_loss = torch::nn::BCELoss(torch::nn::BCELossOptions().reduction(reduction_mode));
        }
        
        // Apply the BCE loss
        torch::Tensor loss = bce_loss->forward(input, target);
        
        // Test backward pass if possible
        if (loss.numel() > 0 && loss.requires_grad()) {
            try {
                loss.backward();
            } catch (...) {
                // Backward pass may fail for various reasons, that's okay
            }
        }
        
        // Test with different reduction modes
        torch::nn::BCELoss mean_loss(torch::nn::BCELossOptions().reduction(torch::kMean));
        torch::nn::BCELoss sum_loss(torch::nn::BCELossOptions().reduction(torch::kSum));
        torch::nn::BCELoss none_loss(torch::nn::BCELossOptions().reduction(torch::kNone));
        
        torch::Tensor mean_result = mean_loss->forward(input, target);
        torch::Tensor sum_result = sum_loss->forward(input, target);
        torch::Tensor none_result = none_loss->forward(input, target);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}