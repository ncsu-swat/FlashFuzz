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
        
        // Create input and target tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // If we have more data, create target tensor
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, create a target with same shape as input
            target = input.clone();
        }
        
        // Get reduction mode from remaining data
        torch::nn::SmoothL1LossOptions::reduction_t reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_selector = Data[offset++] % 3;
            if (reduction_selector == 0) {
                reduction_mode = torch::kNone;
            } else if (reduction_selector == 1) {
                reduction_mode = torch::kMean;
            } else {
                reduction_mode = torch::kSum;
            }
        }
        
        // Get beta value from remaining data
        double beta = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure beta is positive (as required by the API)
            beta = std::abs(beta);
            
            // Avoid extremely large values that might cause numerical issues
            if (beta > 1e10) {
                beta = 1e10;
            }
            
            // Avoid extremely small values that might cause numerical issues
            if (beta < 1e-10 && beta != 0.0) {
                beta = 1e-10;
            }
        }
        
        // Create SmoothL1Loss module with the options
        torch::nn::SmoothL1LossOptions options;
        options.reduction(reduction_mode).beta(beta);
        torch::nn::SmoothL1Loss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor loss = loss_fn(input, target);
        
        // Try backward pass if tensors require grad
        if (offset < Size && Data[offset++] % 2 == 0) {
            auto input_requires_grad = input.clone().set_requires_grad(true);
            auto target_requires_grad = target.clone().set_requires_grad(true);
            
            auto loss_with_grad = loss_fn(input_requires_grad, target_requires_grad);
            
            // Try backward with different grad_output shapes
            if (loss_with_grad.numel() > 0) {
                if (std::holds_alternative<torch::enumtype::kNone>(reduction_mode)) {
                    // For 'none' reduction, create a grad_output with the same shape as the loss
                    auto grad_output = torch::ones_like(loss_with_grad);
                    loss_with_grad.backward(grad_output);
                } else {
                    // For 'mean' or 'sum' reduction, the loss is a scalar
                    loss_with_grad.backward();
                }
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