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
        
        // Create input tensor (log probabilities)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (probabilities)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a tensor with the same shape as input
            target = torch::ones_like(input);
        }
        
        // Parse reduction mode from the input data
        torch::nn::KLDivLossOptions::reduction_t reduction_mode = torch::kMean;
        int64_t functional_reduction = at::Reduction::Mean;
        if (offset < Size) {
            uint8_t reduction_selector = Data[offset++] % 3;
            switch (reduction_selector) {
                case 0:
                    reduction_mode = torch::kNone;
                    functional_reduction = at::Reduction::None;
                    break;
                case 1:
                    reduction_mode = torch::kMean;
                    functional_reduction = at::Reduction::Mean;
                    break;
                case 2:
                    reduction_mode = torch::kSum;
                    functional_reduction = at::Reduction::Sum;
                    break;
            }
        }
        
        // Parse log_target flag
        bool log_target = false;
        if (offset < Size) {
            log_target = (Data[offset++] % 2) == 1;
        }
        
        // Create KLDivLoss module with different options
        torch::nn::KLDivLossOptions options;
        options.reduction(reduction_mode).log_target(log_target);
        torch::nn::KLDivLoss kl_div_loss(options);
        
        // Apply the KL divergence loss
        torch::Tensor loss = kl_div_loss->forward(input, target);
        
        // Test with default options
        torch::nn::KLDivLoss default_kl_div_loss;
        torch::Tensor default_loss = default_kl_div_loss->forward(input, target);
        
        // Test with functional API
        torch::Tensor functional_loss = torch::kl_div(input, target, functional_reduction, log_target);
        
        // Test with different reduction modes
        torch::nn::KLDivLossOptions none_options;
        none_options.reduction(torch::kNone).log_target(log_target);
        torch::nn::KLDivLoss none_kl_div_loss(none_options);
        torch::Tensor none_loss = none_kl_div_loss->forward(input, target);
        
        torch::nn::KLDivLossOptions sum_options;
        sum_options.reduction(torch::kSum).log_target(log_target);
        torch::nn::KLDivLoss sum_kl_div_loss(sum_options);
        torch::Tensor sum_loss = sum_kl_div_loss->forward(input, target);
        
        // Test with log_target=true
        torch::nn::KLDivLossOptions log_target_options;
        log_target_options.reduction(reduction_mode).log_target(true);
        torch::nn::KLDivLoss log_target_kl_div_loss(log_target_options);
        torch::Tensor log_target_loss = log_target_kl_div_loss->forward(input, target);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}