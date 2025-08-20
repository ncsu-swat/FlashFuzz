#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor target;
        
        // Create another tensor for target if there's data left
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no data left, create a target with same shape as input
            target = input.clone();
        }
        
        // Get a loss function type based on remaining data
        uint8_t loss_type = 0;
        if (offset < Size) {
            loss_type = Data[offset++] % 10; // Choose from different loss functions
        }
        
        // Get reduction mode
        torch::Reduction::Reduction reduction = torch::Reduction::Mean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++] % 3;
            switch (reduction_byte) {
                case 0: reduction = torch::Reduction::None; break;
                case 1: reduction = torch::Reduction::Sum; break;
                case 2: reduction = torch::Reduction::Mean; break;
            }
        }
        
        // Get weight parameter for some loss functions
        double weight_param = 0.5;
        if (offset < Size) {
            // Use the next byte to create a weight between 0 and 1
            weight_param = static_cast<double>(Data[offset++]) / 255.0;
        }
        
        // Apply different loss functions
        torch::Tensor loss;
        
        switch (loss_type) {
            case 0: {
                // L1Loss
                auto l1_loss = torch::nn::L1Loss(torch::nn::L1LossOptions().reduction(reduction));
                loss = l1_loss->forward(input, target);
                break;
            }
            case 1: {
                // MSELoss
                auto mse_loss = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(reduction));
                loss = mse_loss->forward(input, target);
                break;
            }
            case 2: {
                // CrossEntropyLoss
                auto ce_loss = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().reduction(reduction));
                loss = ce_loss->forward(input, target);
                break;
            }
            case 3: {
                // BCELoss
                auto bce_loss = torch::nn::BCELoss(torch::nn::BCELossOptions().reduction(reduction));
                // Ensure input and target are in [0,1] range for BCE
                auto sigmoid_input = torch::sigmoid(input);
                auto clamped_target = torch::clamp(target, 0.0, 1.0);
                loss = bce_loss->forward(sigmoid_input, clamped_target);
                break;
            }
            case 4: {
                // BCEWithLogitsLoss
                auto bce_with_logits_loss = torch::nn::BCEWithLogitsLoss(torch::nn::BCEWithLogitsLossOptions().reduction(reduction));
                auto clamped_target = torch::clamp(target, 0.0, 1.0);
                loss = bce_with_logits_loss->forward(input, clamped_target);
                break;
            }
            case 5: {
                // KLDivLoss
                auto kl_div_loss = torch::nn::KLDivLoss(torch::nn::KLDivLossOptions().reduction(reduction));
                auto log_input = torch::log_softmax(input, -1);
                auto softmax_target = torch::softmax(target, -1);
                loss = kl_div_loss->forward(log_input, softmax_target);
                break;
            }
            case 6: {
                // HingeEmbeddingLoss
                auto hinge_loss = torch::nn::HingeEmbeddingLoss(torch::nn::HingeEmbeddingLossOptions().margin(weight_param).reduction(reduction));
                loss = hinge_loss->forward(input, target);
                break;
            }
            case 7: {
                // HuberLoss
                auto huber_loss = torch::nn::HuberLoss(torch::nn::HuberLossOptions().delta(weight_param).reduction(reduction));
                loss = huber_loss->forward(input, target);
                break;
            }
            case 8: {
                // SmoothL1Loss
                auto smooth_l1_loss = torch::nn::SmoothL1Loss(torch::nn::SmoothL1LossOptions().reduction(reduction));
                loss = smooth_l1_loss->forward(input, target);
                break;
            }
            case 9: {
                // NLLLoss
                auto nll_loss = torch::nn::NLLLoss(torch::nn::NLLLossOptions().reduction(reduction));
                auto log_input = torch::log_softmax(input, -1);
                // Convert target to long for NLLLoss
                auto target_long = target.to(torch::kLong);
                loss = nll_loss->forward(log_input, target_long);
                break;
            }
        }
        
        // Use the loss to prevent it from being optimized away
        auto item = loss.item<float>();
        (void)item;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}