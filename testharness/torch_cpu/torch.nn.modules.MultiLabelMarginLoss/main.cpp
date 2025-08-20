#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, create one with the same shape as input
            target = torch::zeros_like(input);
        }
        
        // Ensure input and target have at least 2 dimensions for MultiLabelMarginLoss
        if (input.dim() < 2) {
            input = input.reshape({1, input.numel()});
        }
        
        if (target.dim() < 2) {
            target = target.reshape({1, target.numel()});
        }
        
        // Ensure target has the same shape as input
        if (input.sizes() != target.sizes()) {
            target = target.reshape_as(input);
        }
        
        // Convert target to Long type as required by MultiLabelMarginLoss
        target = target.to(torch::kLong);
        
        // Create reduction based on the remaining data
        torch::nn::MultiLabelMarginLossOptions::reduction_t reduction;
        if (offset < Size) {
            uint8_t reduction_val = Data[offset++] % 3;
            if (reduction_val == 0) {
                reduction = torch::kNone;
            } else if (reduction_val == 1) {
                reduction = torch::kMean;
            } else {
                reduction = torch::kSum;
            }
        } else {
            reduction = torch::kMean; // Default
        }
        
        // Create the loss module with reduction parameter
        torch::nn::MultiLabelMarginLoss loss_module(
            torch::nn::MultiLabelMarginLossOptions().reduction(reduction)
        );
        
        // Apply the loss function
        torch::Tensor output = loss_module->forward(input, target);
        
        // Try backward pass if possible
        if (output.requires_grad()) {
            output.backward();
        }
        
        // Try with different weight configurations if we have more data
        if (offset + 3 < Size) {
            // Create a weight tensor
            torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure weight has proper shape (should be 1D with size matching number of classes)
            if (weight.dim() > 1) {
                weight = weight.reshape({weight.numel()});
            }
            
            // Create a new loss module with weight
            torch::nn::MultiLabelMarginLoss weighted_loss_module(
                torch::nn::MultiLabelMarginLossOptions().reduction(reduction)
            );
            
            // Apply the weighted loss function
            torch::Tensor weighted_output = weighted_loss_module->forward(input, target);
            
            // Try backward pass if possible
            if (weighted_output.requires_grad()) {
                weighted_output.backward();
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