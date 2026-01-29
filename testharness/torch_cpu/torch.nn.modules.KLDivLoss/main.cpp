#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        } else {
            // If we don't have enough data, create a tensor with the same shape as input
            target = torch::ones_like(input);
        }
        
        // Ensure tensors are float type for KL divergence
        input = input.to(torch::kFloat32);
        target = target.to(torch::kFloat32);
        
        // Ensure target is same shape as input (reshape if needed)
        if (input.sizes() != target.sizes()) {
            try {
                target = target.reshape(input.sizes());
            } catch (...) {
                // If reshape fails, create matching tensor
                target = torch::rand_like(input);
            }
        }
        
        // Parse reduction mode from the input data
        torch::nn::KLDivLossOptions::reduction_t reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_selector = Data[offset++] % 3;
            switch (reduction_selector) {
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
        
        // Parse log_target flag
        bool log_target = false;
        if (offset < Size) {
            log_target = (Data[offset++] % 2) == 1;
        }
        
        // Ensure target values are valid probabilities (positive)
        // For log_target=false, target should be probabilities
        // For log_target=true, target should be log probabilities
        if (!log_target) {
            target = torch::abs(target) + 1e-8;  // Ensure positive
            target = target / target.sum();  // Normalize to probabilities
        }
        
        // Input should be log probabilities
        input = torch::log_softmax(input.flatten(), 0).reshape(input.sizes());
        
        // Inner try-catch for operations that may fail due to shape/value issues
        try {
            // Create KLDivLoss module with different options
            torch::nn::KLDivLossOptions options;
            options.reduction(reduction_mode).log_target(log_target);
            torch::nn::KLDivLoss kl_div_loss(options);
            
            // Apply the KL divergence loss
            torch::Tensor loss = kl_div_loss->forward(input, target);
            
            // Test with default options
            torch::nn::KLDivLoss default_kl_div_loss;
            torch::Tensor default_loss = default_kl_div_loss->forward(input, target);
            
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
            torch::Tensor log_target_tensor = torch::log(target.clamp_min(1e-8));
            torch::nn::KLDivLossOptions log_target_options;
            log_target_options.reduction(reduction_mode).log_target(true);
            torch::nn::KLDivLoss log_target_kl_div_loss(log_target_options);
            torch::Tensor log_target_loss = log_target_kl_div_loss->forward(input, log_target_tensor);
            
            // Test functional API - torch::nn::functional::kl_div
            namespace F = torch::nn::functional;
            auto func_options = F::KLDivFuncOptions().reduction(torch::kMean).log_target(log_target);
            torch::Tensor functional_loss = F::kl_div(input, target, func_options);
        } catch (...) {
            // Silently catch expected failures from shape/value mismatches
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}