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
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor (logits) with gradients enabled
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kFloat32);
        input = input.detach().requires_grad_(true);
        
        // Create target tensor with same shape as input (labels should be in [0, 1])
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kFloat32);
            // Reshape target to match input if needed
            try {
                target = target.reshape_as(input);
            } catch (...) {
                // If reshape fails, create matching tensor
                target = torch::rand_like(input);
            }
        } else {
            target = torch::rand_like(input);  // Random values in [0, 1]
        }
        
        // Ensure target values are between 0 and 1 for BCE
        target = torch::sigmoid(target).detach();
        
        // Parse reduction mode
        torch::nn::BCEWithLogitsLossOptions::reduction_t reduction_mode = torch::kMean;
        int64_t reduction_int = at::Reduction::Mean;
        if (offset < Size) {
            uint8_t reduction_selector = Data[offset++] % 3;
            switch (reduction_selector) {
                case 0: 
                    reduction_mode = torch::kNone; 
                    reduction_int = at::Reduction::None;
                    break;
                case 1: 
                    reduction_mode = torch::kMean; 
                    reduction_int = at::Reduction::Mean;
                    break;
                case 2: 
                    reduction_mode = torch::kSum; 
                    reduction_int = at::Reduction::Sum;
                    break;
            }
        }
        
        // Create weight tensor (optional) - must broadcast with input
        torch::Tensor weight;
        bool use_weight = false;
        if (offset < Size) {
            use_weight = Data[offset++] % 2 == 0;
            if (use_weight) {
                // Weight should be broadcastable with input
                // Use last dimension size or scalar
                if (input.dim() > 0) {
                    weight = torch::rand({input.size(-1)}, torch::kFloat32).abs() + 0.1f;
                } else {
                    weight = torch::rand({1}, torch::kFloat32).abs() + 0.1f;
                }
            }
        }
        
        // Create pos_weight tensor (optional) - must broadcast with target
        torch::Tensor pos_weight;
        bool use_pos_weight = false;
        if (offset < Size) {
            use_pos_weight = Data[offset++] % 2 == 0;
            if (use_pos_weight) {
                // pos_weight should be broadcastable with target
                if (input.dim() > 0) {
                    pos_weight = torch::rand({input.size(-1)}, torch::kFloat32).abs() + 0.1f;
                } else {
                    pos_weight = torch::rand({1}, torch::kFloat32).abs() + 0.1f;
                }
            }
        }
        
        // Test 1: BCEWithLogitsLoss module with various options
        {
            torch::nn::BCEWithLogitsLossOptions options;
            options.reduction(reduction_mode);
            
            if (use_weight && weight.defined()) {
                options.weight(weight);
            }
            
            if (use_pos_weight && pos_weight.defined()) {
                options.pos_weight(pos_weight);
            }
            
            torch::nn::BCEWithLogitsLoss bce_loss(options);
            
            torch::Tensor loss = bce_loss(input, target);
            
            // Try backward pass if loss is scalar and requires grad
            if (loss.numel() == 1 && input.requires_grad()) {
                loss.backward();
            }
        }
        
        // Test 2: Functional version
        {
            torch::Tensor input2 = input.detach().requires_grad_(true);
            
            std::optional<torch::Tensor> weight_opt = use_weight && weight.defined() 
                ? std::optional<torch::Tensor>(weight) 
                : std::nullopt;
            std::optional<torch::Tensor> pos_weight_opt = use_pos_weight && pos_weight.defined() 
                ? std::optional<torch::Tensor>(pos_weight) 
                : std::nullopt;
            
            torch::Tensor functional_loss = at::binary_cross_entropy_with_logits(
                input2, target, weight_opt, pos_weight_opt, reduction_int);
            
            // Backward on functional version
            if (functional_loss.numel() == 1 && input2.requires_grad()) {
                functional_loss.backward();
            }
        }
        
        // Test 3: Different input configurations
        if (offset < Size && input.numel() > 0) {
            // Test with batched input
            try {
                torch::Tensor batched_input = input.unsqueeze(0).repeat({2, 1});
                batched_input = batched_input.detach().requires_grad_(true);
                torch::Tensor batched_target = target.unsqueeze(0).repeat({2, 1});
                
                torch::nn::BCEWithLogitsLoss default_loss;
                torch::Tensor batch_loss = default_loss(batched_input, batched_target);
            } catch (...) {
                // Shape manipulation may fail for some inputs
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