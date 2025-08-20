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
        
        // Create input tensor (predictions)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (ground truth)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, clone the first one
            target = input.clone();
        }
        
        // Ensure input and target have the same shape
        if (input.sizes() != target.sizes()) {
            if (input.dim() > 0 && target.dim() > 0) {
                // Reshape target to match input
                target = target.reshape_as(input);
            }
        }
        
        // Ensure input and target values are between 0 and 1 (required for BCE)
        input = torch::sigmoid(input);
        target = torch::sigmoid(target);
        
        // Extract reduction mode from data if available
        torch::enumtype::Reduction reduction = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0: reduction = torch::kNone; break;
                case 1: reduction = torch::kSum; break;
                case 2: reduction = torch::kMean; break;
            }
        }
        
        // Extract weight tensor if there's data left
        torch::Tensor weight;
        bool use_weight = false;
        if (offset < Size) {
            uint8_t use_weight_byte = Data[offset++];
            use_weight = (use_weight_byte % 2 == 1);
            
            if (use_weight && offset < Size) {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
                if (weight.sizes() != input.sizes()) {
                    if (weight.dim() > 0) {
                        // Reshape weight to match input
                        weight = weight.reshape_as(input);
                    }
                }
            }
        }
        
        // Create BCE Loss module with different configurations
        torch::nn::BCELoss bce_loss;
        
        // Configure the BCE Loss based on available data
        if (use_weight && weight.defined()) {
            bce_loss = torch::nn::BCELoss(torch::nn::BCELossOptions().weight(weight).reduction(reduction));
        } else {
            bce_loss = torch::nn::BCELoss(torch::nn::BCELossOptions().reduction(reduction));
        }
        
        // Apply the BCE Loss
        torch::Tensor loss = bce_loss->forward(input, target);
        
        // Try backward pass if loss is scalar or has gradient
        if (loss.requires_grad() || (loss.dim() == 0 && input.requires_grad())) {
            try {
                if (loss.dim() == 0) {
                    loss.backward();
                } else {
                    // For "none" reduction, we need to sum the loss
                    loss.sum().backward();
                }
            } catch (...) {
                // Ignore backward pass errors
            }
        }
        
        // Try with different options if there's more data
        if (offset + 1 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Try with different reduction
            torch::enumtype::Reduction alt_reduction;
            switch (option_byte % 3) {
                case 0: alt_reduction = torch::kNone; break;
                case 1: alt_reduction = torch::kSum; break;
                case 2: alt_reduction = torch::kMean; break;
            }
            
            if (alt_reduction != reduction) {
                torch::nn::BCELoss alt_bce_loss(torch::nn::BCELossOptions().reduction(alt_reduction));
                torch::Tensor alt_loss = alt_bce_loss->forward(input, target);
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