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
        
        // Need at least some data to create tensors and config
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor with gradients enabled for backward pass testing
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        input = input.to(torch::kFloat32).requires_grad_(true);
        
        // Create target tensor with same shape as input
        // SoftMarginLoss expects targets to be -1 or +1
        torch::Tensor target_raw = fuzzer_utils::createTensor(Data, Size, offset);
        // Ensure target has same shape as input and values are -1 or +1
        target_raw = target_raw.to(torch::kFloat32);
        
        // Reshape target to match input if needed, then convert to -1/+1
        torch::Tensor target;
        try {
            if (target_raw.numel() >= input.numel()) {
                target = target_raw.flatten().slice(0, 0, input.numel()).view(input.sizes());
            } else {
                // Repeat target to match input size
                int64_t repeats = (input.numel() / target_raw.numel()) + 1;
                target = target_raw.flatten().repeat({repeats}).slice(0, 0, input.numel()).view(input.sizes());
            }
            // Convert to -1 or +1 based on sign
            target = torch::where(target >= 0, torch::ones_like(target), -torch::ones_like(target));
        } catch (...) {
            // If shape manipulation fails, create matching tensor directly
            target = torch::ones_like(input);
            if (offset < Size && Data[offset] % 2 == 0) {
                target = -target;
            }
        }
        
        // Parse configuration byte for reduction mode
        uint8_t config_byte = 0;
        if (offset < Size) {
            config_byte = Data[offset++];
        }
        
        // Track reduction mode as integer for comparison purposes
        int reduction_mode_int = config_byte % 3;
        
        // Create SoftMarginLoss module with appropriate reduction
        torch::nn::SoftMarginLossOptions options;
        switch (reduction_mode_int) {
            case 0:
                options.reduction(torch::kNone);
                break;
            case 1:
                options.reduction(torch::kMean);
                break;
            case 2:
                options.reduction(torch::kSum);
                break;
            default:
                options.reduction(torch::kMean);
                reduction_mode_int = 1;
        }
        
        torch::nn::SoftMarginLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor output = loss_fn->forward(input, target);
        
        // Test backward pass only if output is scalar (kMean or kSum reduction)
        if (reduction_mode_int != 0 && output.numel() == 1) {
            try {
                output.backward();
            } catch (...) {
                // Backward may fail for various reasons, that's okay
            }
        }
        
        // Test with a different reduction mode
        if (offset + 2 < Size) {
            // Create fresh input for second test to avoid gradient graph issues
            torch::Tensor input2 = input.detach().clone().requires_grad_(true);
            
            torch::nn::SoftMarginLossOptions other_options;
            int other_reduction = Data[offset++] % 3;
            if (other_reduction == 0) {
                other_options.reduction(torch::kNone);
            } else if (other_reduction == 1) {
                other_options.reduction(torch::kMean);
            } else {
                other_options.reduction(torch::kSum);
            }
            
            torch::nn::SoftMarginLoss other_loss_fn(other_options);
            torch::Tensor other_output = other_loss_fn->forward(input2, target);
            
            // Test backward on second output if scalar
            if (other_reduction != 0 && other_output.numel() == 1) {
                try {
                    other_output.backward();
                } catch (...) {
                    // Backward may fail, that's okay
                }
            }
        }
        
        // Test functional interface as well
        try {
            torch::Tensor input3 = input.detach().clone();
            torch::Tensor func_output = torch::nn::functional::soft_margin_loss(
                input3, target, 
                torch::nn::functional::SoftMarginLossFuncOptions().reduction(torch::kMean));
        } catch (...) {
            // Functional API might have different requirements
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}