#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor (logits)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (labels)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, create one with same shape as input
            target = torch::zeros_like(input);
        }
        
        // Create weight tensor (optional)
        torch::Tensor weight;
        bool use_weight = false;
        if (offset < Size) {
            use_weight = Data[offset++] % 2 == 0;
            if (use_weight && offset < Size) {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
            }
        }
        
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
        
        // Parse pos_weight parameter (optional)
        torch::Tensor pos_weight;
        bool use_pos_weight = false;
        if (offset < Size) {
            use_pos_weight = Data[offset++] % 2 == 0;
            if (use_pos_weight && offset < Size) {
                pos_weight = fuzzer_utils::createTensor(Data, Size, offset);
            }
        }
        
        // Create BCEWithLogitsLoss module with various options
        torch::nn::BCEWithLogitsLossOptions options;
        options.reduction(reduction_mode);
        
        if (use_weight && weight.defined()) {
            options.weight(weight);
        }
        
        if (use_pos_weight && pos_weight.defined()) {
            options.pos_weight(pos_weight);
        }
        
        torch::nn::BCEWithLogitsLoss bce_loss(options);
        
        // Try to make target values between 0 and 1 for BCE
        if (target.defined()) {
            target = torch::sigmoid(target);
        }
        
        // Apply the loss function
        torch::Tensor loss = bce_loss(input, target);
        
        // Try backward pass if loss is scalar
        if (loss.numel() == 1 && loss.requires_grad()) {
            loss.backward();
        }
        
        // Test functional version as well
        std::optional<torch::Tensor> weight_opt = use_weight && weight.defined() ? std::optional<torch::Tensor>(weight) : std::nullopt;
        std::optional<torch::Tensor> pos_weight_opt = use_pos_weight && pos_weight.defined() ? std::optional<torch::Tensor>(pos_weight) : std::nullopt;
        
        torch::Tensor functional_loss = at::binary_cross_entropy_with_logits(
            input, target, weight_opt, pos_weight_opt, reduction_int);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
