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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input and target tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the target tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get reduction mode from the input data
        torch::nn::MSELossOptions::reduction_t reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0:
                    reduction_mode = torch::kMean;
                    break;
                case 1:
                    reduction_mode = torch::kSum;
                    break;
                case 2:
                    reduction_mode = torch::kNone;
                    break;
            }
        }
        
        // Inner try-catch for expected failures (shape mismatches, etc.)
        try {
            // Create MSELoss module with the selected reduction mode
            torch::nn::MSELoss mse_loss(torch::nn::MSELossOptions().reduction(reduction_mode));
            
            // Ensure input and target have the same shape for MSELoss
            // If shapes don't match, try to broadcast or reshape
            torch::Tensor input_f = input.to(torch::kFloat);
            torch::Tensor target_f = target.to(torch::kFloat);
            
            // Apply MSELoss to the input and target tensors
            torch::Tensor loss = mse_loss(input_f, target_f);
            
            // Try to access the loss value to ensure computation is performed
            if (loss.defined() && loss.numel() > 0) {
                if (loss.numel() == 1) {
                    float loss_value = loss.item<float>();
                    (void)loss_value;
                } else {
                    // For kNone reduction, loss has same shape as input
                    float first_value = loss.flatten()[0].item<float>();
                    (void)first_value;
                }
            }
            
            // Try backward pass if tensors require grad
            if (offset < Size && Data[offset] % 2 == 0) {
                // Create new tensors with requires_grad=true
                torch::Tensor input_grad = input_f.detach().clone().requires_grad_(true);
                torch::Tensor target_grad = target_f.detach().clone();
                
                // Recompute loss with gradients
                torch::Tensor loss_grad = mse_loss(input_grad, target_grad);
                
                // Perform backward pass
                if (loss_grad.numel() == 1) {
                    loss_grad.backward();
                } else if (loss_grad.numel() > 0) {
                    // For non-scalar loss, need a gradient tensor
                    loss_grad.sum().backward();
                }
                
                // Access gradients to ensure they were computed
                if (input_grad.grad().defined()) {
                    float grad_sum = input_grad.grad().sum().item<float>();
                    (void)grad_sum;
                }
            }
        } catch (const c10::Error& e) {
            // Expected failures (shape mismatch, broadcasting issues, etc.)
            // Silently catch - these are valid exploration paths
        } catch (const std::runtime_error& e) {
            // Other runtime errors that may occur during tensor operations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}