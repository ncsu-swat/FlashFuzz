#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create log probabilities tensor (input)
        torch::Tensor log_probs = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create targets tensor (labels)
        torch::Tensor targets;
        if (offset < Size) {
            targets = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create input lengths tensor
        torch::Tensor input_lengths;
        if (offset < Size) {
            input_lengths = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create target lengths tensor
        torch::Tensor target_lengths;
        if (offset < Size) {
            target_lengths = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Get reduction type from input data
        torch::nn::CTCLossOptions::reduction_t reduction = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0:
                    reduction = torch::kNone;
                    break;
                case 1:
                    reduction = torch::kMean;
                    break;
                case 2:
                    reduction = torch::kSum;
                    break;
            }
        }
        
        // Get zero_infinity flag from input data
        bool zero_infinity = false;
        if (offset < Size) {
            zero_infinity = Data[offset++] & 0x1;
        }
        
        // Create CTCLoss module with different options
        torch::nn::CTCLossOptions options;
        options.reduction(reduction).zero_infinity(zero_infinity);
        
        torch::nn::CTCLoss ctc_loss(options);
        
        // Try to convert tensors to appropriate types if needed
        try {
            // Log probs should be float for numerical stability
            if (log_probs.scalar_type() != torch::kFloat && 
                log_probs.scalar_type() != torch::kDouble) {
                log_probs = log_probs.to(torch::kFloat);
            }
            
            // Targets should be integers
            if (!torch::isIntegralType(targets.scalar_type(), true)) {
                targets = targets.to(torch::kLong);
            }
            
            // Input lengths should be integers
            if (!torch::isIntegralType(input_lengths.scalar_type(), true)) {
                input_lengths = input_lengths.to(torch::kLong);
            }
            
            // Target lengths should be integers
            if (!torch::isIntegralType(target_lengths.scalar_type(), true)) {
                target_lengths = target_lengths.to(torch::kLong);
            }
        } catch (const std::exception& e) {
            // If conversion fails, just continue with original tensors
        }
        
        // Apply CTCLoss
        torch::Tensor loss = ctc_loss(log_probs, targets, input_lengths, target_lengths);
        
        // Try backward pass if possible
        if (loss.numel() > 0 && loss.requires_grad()) {
            try {
                loss.backward();
            } catch (const std::exception& e) {
                // Ignore backward pass errors
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}