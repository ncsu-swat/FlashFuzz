#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least some data to create tensors and configure options
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the target tensor
        if (offset >= Size) {
            return 0;
        }
        
        // Create target tensor with the same shape as input for valid MSELoss computation
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get reduction mode from the remaining data
        torch::nn::MSELossOptions options;
        uint8_t reduction_selector = 1; // Default to mean
        if (offset < Size) {
            reduction_selector = Data[offset++] % 3;
        }
        
        switch (reduction_selector) {
            case 0:
                options.reduction(torch::kNone);
                break;
            case 1:
                options.reduction(torch::kMean);
                break;
            case 2:
                options.reduction(torch::kSum);
                break;
        }
        
        // Create MSELoss module with the selected reduction mode
        torch::nn::MSELoss mse_loss(options);
        
        // Inner try-catch for expected shape mismatch errors
        try {
            // Apply MSELoss to the input and target tensors
            torch::Tensor loss = mse_loss(input, target);
            
            // Ensure computation is completed
            // Only call item() if the result is a scalar (reduction != None)
            if (reduction_selector != 0) {
                (void)loss.item<float>();
            } else {
                // For reduction=None, just access the data to ensure computation
                (void)loss.sum().item<float>();
            }
            
            // Try backward pass if tensors require grad
            if (offset < Size && Data[offset++] % 2 == 0) {
                // Create new tensors with requires_grad=true
                torch::Tensor input_with_grad = input.detach().clone().set_requires_grad(true);
                torch::Tensor target_with_grad = target.detach().clone();
                
                // Recompute loss with gradients
                torch::Tensor loss_with_grad = mse_loss(input_with_grad, target_with_grad);
                
                // Backward pass - need a scalar for backward
                if (reduction_selector == 0) {
                    loss_with_grad.sum().backward();
                } else {
                    loss_with_grad.backward();
                }
                
                // Access gradients to ensure they're computed
                (void)input_with_grad.grad();
            }
        }
        catch (const c10::Error&) {
            // Expected errors (shape mismatch, etc.) - silently ignore
        }
        catch (const std::runtime_error&) {
            // Expected runtime errors - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}