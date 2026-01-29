#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have enough data for the target tensor
        if (offset >= Size) {
            return 0;
        }
        
        // Create target tensor with the SAME shape as input
        // This is required for SmoothL1Loss
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape target to match input shape if they differ
        try {
            if (!input.sizes().equals(target.sizes())) {
                target = target.view({-1}).narrow(0, 0, std::min(target.numel(), input.numel()));
                target = target.expand_as(input.view({-1})).view(input.sizes());
            }
        } catch (...) {
            // If reshape fails, create target with same shape
            target = torch::rand_like(input);
        }
        
        // Get reduction mode from the input data
        torch::nn::SmoothL1LossOptions::reduction_t reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0:
                    reduction_mode = torch::kNone;
                    break;
                case 1:
                    reduction_mode = torch::kMean;
                    break;
                case 2:
                    reduction_mode = torch::kSum;
                    break;
            }
        }
        
        // Get beta value from the input data
        double beta = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Handle NaN and Inf
            if (std::isnan(beta) || std::isinf(beta)) {
                beta = 1.0;
            }
            
            // Ensure beta is positive (as required by SmoothL1Loss)
            beta = std::abs(beta);
            
            // Avoid extremely large values that might cause numerical issues
            if (beta > 1e10) {
                beta = 1e10;
            }
            
            // Avoid extremely small values that might cause numerical issues
            if (beta < 1e-10) {
                beta = 1e-10;
            }
        }
        
        // Create SmoothL1Loss module with the extracted options
        torch::nn::SmoothL1LossOptions options;
        options.reduction(reduction_mode).beta(beta);
        torch::nn::SmoothL1Loss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor loss = loss_fn(input, target);
        
        // Force computation - use sum() to handle both scalar and tensor outputs
        (void)loss.sum().item<float>();
        
        // Test with different beta values
        if (offset < Size) {
            double beta2 = 0.5;
            torch::nn::SmoothL1LossOptions options2;
            options2.reduction(torch::kMean).beta(beta2);
            torch::nn::SmoothL1Loss loss_fn2(options2);
            torch::Tensor loss2 = loss_fn2(input, target);
            (void)loss2.sum().item<float>();
            
            // Try with kSum reduction
            torch::nn::SmoothL1LossOptions options3;
            options3.reduction(torch::kSum).beta(beta);
            torch::nn::SmoothL1Loss loss_fn3(options3);
            torch::Tensor loss3 = loss_fn3(input, target);
            (void)loss3.sum().item<float>();
        }
        
        // Test the functional version
        torch::Tensor functional_loss = torch::nn::functional::smooth_l1_loss(
            input, target, 
            torch::nn::functional::SmoothL1LossFuncOptions()
                .reduction(reduction_mode)
                .beta(beta));
        
        (void)functional_loss.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}