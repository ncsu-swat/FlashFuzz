#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least some data to create tensors and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor with requires_grad for backward pass testing
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is float type for loss computation
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure input is at least 2D (batch_size, num_classes)
        if (input.dim() < 2) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 2) {
            input = input.unsqueeze(1);
        }
        
        // Enable gradient computation
        input = input.clone().detach().requires_grad_(true);
        
        // Create target tensor with the same shape as input
        torch::Tensor target = torch::rand_like(input).round();  // Binary 0/1 values
        
        // Parse reduction mode from the input data
        torch::nn::MultiLabelSoftMarginLossOptions options;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            int reduction_mode = reduction_byte % 3;
            
            switch (reduction_mode) {
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
        }
        
        // Parse weight tensor option
        if (offset < Size) {
            bool use_weight = Data[offset++] % 2;
            if (use_weight && input.dim() >= 2) {
                int64_t num_classes = input.size(-1);
                if (num_classes > 0 && num_classes <= 1024) {
                    torch::Tensor weight = torch::rand({num_classes}, torch::kFloat32).abs() + 0.1f;
                    options.weight(weight);
                }
            }
        }
        
        // Create the loss function
        torch::nn::MultiLabelSoftMarginLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor output;
        try {
            output = loss_fn(input, target);
        } catch (const c10::Error&) {
            // Shape mismatch or other expected errors
            return 0;
        }
        
        // Verify output is valid
        if (output.numel() == 0) {
            return 0;
        }
        
        // Access output to ensure computation happened
        if (output.dim() == 0) {
            // Scalar output (mean or sum reduction)
            volatile float loss_value = output.item<float>();
            (void)loss_value;
        } else {
            // Non-scalar output (none reduction)
            volatile float first_val = output.flatten()[0].item<float>();
            (void)first_val;
        }
        
        // Test backward pass
        try {
            torch::Tensor loss_for_backward = output;
            if (output.dim() > 0) {
                loss_for_backward = output.sum();
            }
            loss_for_backward.backward();
            
            // Access gradient to ensure backward completed
            if (input.grad().defined()) {
                volatile float grad_sum = input.grad().sum().item<float>();
                (void)grad_sum;
            }
        } catch (const c10::Error&) {
            // Backward pass can fail for various reasons, not a bug
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}