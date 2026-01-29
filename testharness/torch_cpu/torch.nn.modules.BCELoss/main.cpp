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
        
        // Create input tensor (predictions) with requires_grad for backward testing
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (ground truth)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            target = input.clone();
        }
        
        // Ensure input and target have the same shape
        if (input.sizes() != target.sizes() && input.numel() > 0 && target.numel() > 0) {
            try {
                target = target.reshape_as(input);
            } catch (...) {
                target = torch::rand_like(input);
            }
        }
        
        // Ensure input values are between 0 and 1 for BCE (with requires_grad for backward)
        if (input.is_floating_point()) {
            input = torch::sigmoid(input.detach().requires_grad_(true));
        } else {
            input = torch::sigmoid(input.to(torch::kFloat).detach().requires_grad_(true));
        }
        
        // Ensure target values are between 0 and 1 for BCE (no grad needed for target)
        if (target.is_floating_point()) {
            target = torch::clamp(target.detach(), 0.0, 1.0);
        } else {
            target = torch::clamp(target.to(torch::kFloat).detach(), 0.0, 1.0);
        }
        
        // Parse reduction mode from input data
        // Track with an int to avoid variant comparison issues
        int reduction_type = 0; // 0=mean, 1=sum, 2=none
        torch::nn::BCELossOptions::reduction_t reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0:
                    reduction_mode = torch::kMean;
                    reduction_type = 0;
                    break;
                case 1:
                    reduction_mode = torch::kSum;
                    reduction_type = 1;
                    break;
                case 2:
                    reduction_mode = torch::kNone;
                    reduction_type = 2;
                    break;
            }
        }
        
        // Create BCE Loss module
        torch::nn::BCELoss bce_loss(torch::nn::BCELossOptions().reduction(reduction_mode));
        
        // Apply the BCE loss
        torch::Tensor loss = bce_loss->forward(input, target);
        
        // Test backward pass
        if (loss.numel() > 0) {
            try {
                if (reduction_type == 2) {
                    // For 'none' reduction, we need to sum before backward
                    loss.sum().backward();
                } else {
                    loss.backward();
                }
            } catch (...) {
                // Backward pass may fail for various reasons
            }
        }
        
        // Also test the functional interface for better coverage
        try {
            torch::Tensor func_loss = torch::nn::functional::binary_cross_entropy(
                input.detach().requires_grad_(true), 
                target,
                torch::nn::functional::BinaryCrossEntropyFuncOptions().reduction(reduction_mode)
            );
        } catch (...) {
            // Functional interface might have different constraints
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}