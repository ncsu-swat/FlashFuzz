#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Get a loss function type based on data
        uint8_t loss_type = Data[offset++] % 10;
        
        // Get reduction mode
        uint8_t reduction_byte = Data[offset++] % 3;
        
        // Get weight parameter for some loss functions (ensure positive)
        double weight_param = 0.1 + (static_cast<double>(Data[offset++]) / 255.0) * 0.9;
        
        // Create batch size and num_classes from fuzzer data
        int batch_size = 1 + (Data[offset++] % 8);
        int num_classes = 2 + (Data[offset++] % 10);
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        if (input.numel() == 0) {
            return 0;
        }
        
        torch::Tensor loss;
        
        switch (loss_type) {
            case 0: {
                // L1Loss - input and target must have same shape
                torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
                if (target.numel() == 0) {
                    target = torch::randn_like(input);
                } else if (!target.sizes().equals(input.sizes())) {
                    target = torch::randn_like(input);
                }
                torch::nn::L1LossOptions opts;
                if (reduction_byte == 0) opts.reduction(torch::kNone);
                else if (reduction_byte == 1) opts.reduction(torch::kSum);
                else opts.reduction(torch::kMean);
                auto l1_loss = torch::nn::L1Loss(opts);
                loss = l1_loss->forward(input, target);
                break;
            }
            case 1: {
                // MSELoss - input and target must have same shape
                torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
                if (target.numel() == 0 || !target.sizes().equals(input.sizes())) {
                    target = torch::randn_like(input);
                }
                torch::nn::MSELossOptions opts;
                if (reduction_byte == 0) opts.reduction(torch::kNone);
                else if (reduction_byte == 1) opts.reduction(torch::kSum);
                else opts.reduction(torch::kMean);
                auto mse_loss = torch::nn::MSELoss(opts);
                loss = mse_loss->forward(input, target);
                break;
            }
            case 2: {
                // CrossEntropyLoss - input: [N, C], target: [N] with class indices
                auto ce_input = torch::randn({batch_size, num_classes});
                auto ce_target = torch::randint(0, num_classes, {batch_size});
                torch::nn::CrossEntropyLossOptions opts;
                if (reduction_byte == 0) opts.reduction(torch::kNone);
                else if (reduction_byte == 1) opts.reduction(torch::kSum);
                else opts.reduction(torch::kMean);
                auto ce_loss = torch::nn::CrossEntropyLoss(opts);
                loss = ce_loss->forward(ce_input, ce_target);
                break;
            }
            case 3: {
                // BCELoss - input and target must be in [0,1] and same shape
                auto sigmoid_input = torch::sigmoid(input);
                auto clamped_target = torch::rand_like(input);
                torch::nn::BCELossOptions opts;
                if (reduction_byte == 0) opts.reduction(torch::kNone);
                else if (reduction_byte == 1) opts.reduction(torch::kSum);
                else opts.reduction(torch::kMean);
                auto bce_loss = torch::nn::BCELoss(opts);
                loss = bce_loss->forward(sigmoid_input, clamped_target);
                break;
            }
            case 4: {
                // BCEWithLogitsLoss - target must be in [0,1], same shape as input
                auto clamped_target = torch::rand_like(input);
                torch::nn::BCEWithLogitsLossOptions opts;
                if (reduction_byte == 0) opts.reduction(torch::kNone);
                else if (reduction_byte == 1) opts.reduction(torch::kSum);
                else opts.reduction(torch::kMean);
                auto bce_with_logits_loss = torch::nn::BCEWithLogitsLoss(opts);
                loss = bce_with_logits_loss->forward(input, clamped_target);
                break;
            }
            case 5: {
                // KLDivLoss - needs proper probability distributions
                auto kl_input = torch::randn({batch_size, num_classes});
                auto kl_target = torch::randn({batch_size, num_classes});
                auto log_input = torch::log_softmax(kl_input, -1);
                auto softmax_target = torch::softmax(kl_target, -1);
                torch::nn::KLDivLossOptions opts;
                if (reduction_byte == 0) opts.reduction(torch::kNone);
                else if (reduction_byte == 1) opts.reduction(torch::kSum);
                else opts.reduction(torch::kMean);
                auto kl_div_loss = torch::nn::KLDivLoss(opts);
                loss = kl_div_loss->forward(log_input, softmax_target);
                break;
            }
            case 6: {
                // HingeEmbeddingLoss - target should be +1 or -1
                auto hinge_target = torch::where(torch::rand_like(input) > 0.5, 
                                                  torch::ones_like(input), 
                                                  -torch::ones_like(input));
                torch::nn::HingeEmbeddingLossOptions opts;
                opts.margin(weight_param);
                if (reduction_byte == 0) opts.reduction(torch::kNone);
                else if (reduction_byte == 1) opts.reduction(torch::kSum);
                else opts.reduction(torch::kMean);
                auto hinge_loss = torch::nn::HingeEmbeddingLoss(opts);
                loss = hinge_loss->forward(input, hinge_target);
                break;
            }
            case 7: {
                // HuberLoss - delta must be positive
                torch::Tensor target = torch::randn_like(input);
                torch::nn::HuberLossOptions opts;
                opts.delta(weight_param);
                if (reduction_byte == 0) opts.reduction(torch::kNone);
                else if (reduction_byte == 1) opts.reduction(torch::kSum);
                else opts.reduction(torch::kMean);
                auto huber_loss = torch::nn::HuberLoss(opts);
                loss = huber_loss->forward(input, target);
                break;
            }
            case 8: {
                // SmoothL1Loss
                torch::Tensor target = torch::randn_like(input);
                torch::nn::SmoothL1LossOptions opts;
                opts.beta(weight_param);
                if (reduction_byte == 0) opts.reduction(torch::kNone);
                else if (reduction_byte == 1) opts.reduction(torch::kSum);
                else opts.reduction(torch::kMean);
                auto smooth_l1_loss = torch::nn::SmoothL1Loss(opts);
                loss = smooth_l1_loss->forward(input, target);
                break;
            }
            case 9: {
                // NLLLoss - input: [N, C] log probabilities, target: [N] class indices
                auto nll_input = torch::log_softmax(torch::randn({batch_size, num_classes}), -1);
                auto nll_target = torch::randint(0, num_classes, {batch_size});
                torch::nn::NLLLossOptions opts;
                if (reduction_byte == 0) opts.reduction(torch::kNone);
                else if (reduction_byte == 1) opts.reduction(torch::kSum);
                else opts.reduction(torch::kMean);
                auto nll_loss = torch::nn::NLLLoss(opts);
                loss = nll_loss->forward(nll_input, nll_target);
                break;
            }
        }
        
        // Use the loss to prevent optimization
        if (loss.defined() && loss.numel() > 0) {
            try {
                volatile float item = loss.mean().item<float>();
                (void)item;
            } catch (...) {
                // Ignore item extraction errors (e.g., for reduction=None)
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