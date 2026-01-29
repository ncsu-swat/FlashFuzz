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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor (log probabilities)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (probabilities)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure target has the same shape as input for KL divergence
            try {
                target = target.expand_as(input).clone();
            } catch (...) {
                target = torch::rand_like(input);
            }
        } else {
            // If we don't have enough data, create a tensor with the same shape as input
            target = torch::rand_like(input);
        }
        
        // Parse reduction mode from the next byte if available
        torch::nn::KLDivLossOptions::reduction_t reduction = torch::kMean;
        int reduction_mode = 1; // 0=none, 1=mean, 2=sum for functional API
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0: 
                    reduction = torch::kNone; 
                    reduction_mode = 0;
                    break;
                case 1: 
                    reduction = torch::kMean; 
                    reduction_mode = 1;
                    break;
                case 2: 
                    reduction = torch::kSum; 
                    reduction_mode = 2;
                    break;
            }
        }
        
        // Parse log_target flag from the next byte if available
        bool log_target = false;
        if (offset < Size) {
            log_target = (Data[offset++] % 2) == 1;
        }
        
        // Create KLDivLoss module with different configurations
        auto options = torch::nn::KLDivLossOptions().reduction(reduction).log_target(log_target);
        torch::nn::KLDivLoss kl_div_loss(options);
        
        // Apply KLDivLoss
        torch::Tensor loss = kl_div_loss->forward(input, target);
        
        // Try with different reduction modes
        if (offset < Size) {
            uint8_t alt_reduction_byte = Data[offset++];
            torch::nn::KLDivLossOptions::reduction_t alt_reduction;
            switch (alt_reduction_byte % 3) {
                case 0: alt_reduction = torch::kNone; break;
                case 1: alt_reduction = torch::kMean; break;
                case 2: alt_reduction = torch::kSum; break;
            }
            
            auto alt_options = torch::nn::KLDivLossOptions().reduction(alt_reduction).log_target(log_target);
            torch::nn::KLDivLoss alt_kl_div_loss(alt_options);
            torch::Tensor alt_loss = alt_kl_div_loss->forward(input, target);
        }
        
        // Try with different log_target setting
        if (offset < Size) {
            bool alt_log_target = !log_target;
            auto alt_options = torch::nn::KLDivLossOptions().reduction(reduction).log_target(alt_log_target);
            torch::nn::KLDivLoss alt_kl_div_loss(alt_options);
            torch::Tensor alt_loss = alt_kl_div_loss->forward(input, target);
        }
        
        // Try functional version using torch::nn::functional
        try {
            auto func_options = torch::nn::functional::KLDivFuncOptions().reduction(reduction).log_target(log_target);
            torch::Tensor functional_loss = torch::nn::functional::kl_div(input, target, func_options);
        } catch (...) {
            // Functional API might have different behavior, silently catch
        }
        
        // Test with requires_grad for backward pass coverage
        if (offset < Size && (Data[offset++] % 4) == 0) {
            torch::Tensor grad_input = input.clone().detach().requires_grad_(true);
            torch::Tensor grad_target = target.clone().detach();
            
            auto grad_options = torch::nn::KLDivLossOptions().reduction(torch::kMean).log_target(log_target);
            torch::nn::KLDivLoss grad_loss_module(grad_options);
            torch::Tensor grad_loss = grad_loss_module->forward(grad_input, grad_target);
            
            try {
                grad_loss.backward();
            } catch (...) {
                // Backward might fail for certain configurations
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