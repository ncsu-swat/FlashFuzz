#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Parse options first to ensure deterministic tensor creation
        bool full = Data[offset++] % 2 == 0;
        uint8_t red_val = Data[offset++] % 3;
        torch::nn::GaussianNLLLossOptions::reduction_t reduction;
        switch (red_val) {
            case 0: reduction = torch::kNone; break;
            case 1: reduction = torch::kSum; break;
            case 2: 
            default: reduction = torch::kMean; break;
        }
        
        // Determine batch size and feature size from data
        int64_t batch_size = 1 + (Data[offset++] % 8);  // 1-8
        int64_t feature_size = 1 + (Data[offset++] % 8); // 1-8
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        if (!input.defined() || input.numel() == 0) {
            return 0;
        }
        
        // Reshape input to a consistent shape
        int64_t total_elements = batch_size * feature_size;
        input = input.flatten();
        if (input.numel() < total_elements) {
            // Pad with zeros if needed
            torch::Tensor padding = torch::zeros({total_elements - input.numel()}, input.options());
            input = torch::cat({input, padding});
        } else {
            input = input.slice(0, 0, total_elements);
        }
        input = input.reshape({batch_size, feature_size}).to(torch::kFloat32);
        
        // Create target tensor with same shape
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        if (!target.defined() || target.numel() == 0) {
            target = torch::zeros({batch_size, feature_size}, torch::kFloat32);
        } else {
            target = target.flatten();
            if (target.numel() < total_elements) {
                torch::Tensor padding = torch::zeros({total_elements - target.numel()}, target.options());
                target = torch::cat({target, padding});
            } else {
                target = target.slice(0, 0, total_elements);
            }
            target = target.reshape({batch_size, feature_size}).to(torch::kFloat32);
        }
        
        // Create variance tensor - can be same shape as input or just feature_size
        torch::Tensor variance = fuzzer_utils::createTensor(Data, Size, offset);
        if (!variance.defined() || variance.numel() == 0) {
            variance = torch::ones({batch_size, feature_size}, torch::kFloat32);
        } else {
            variance = variance.flatten();
            if (variance.numel() < total_elements) {
                torch::Tensor padding = torch::ones({total_elements - variance.numel()}, variance.options());
                variance = torch::cat({variance, padding});
            } else {
                variance = variance.slice(0, 0, total_elements);
            }
            variance = variance.reshape({batch_size, feature_size}).to(torch::kFloat32);
        }
        
        // Ensure variance is positive (required by GaussianNLLLoss)
        double eps = 1e-6;
        variance = torch::abs(variance) + eps;
        
        // Create the GaussianNLLLoss module
        auto options = torch::nn::GaussianNLLLossOptions()
            .full(full)
            .eps(eps)
            .reduction(reduction);
        torch::nn::GaussianNLLLoss loss_fn(options);
        
        // Apply the Gaussian NLL loss function
        torch::Tensor loss = loss_fn(input, target, variance);
        
        // Access the result to ensure computation happens
        if (loss.defined()) {
            volatile float check = loss.sum().item<float>();
            (void)check;
        }
        
        // Test with requires_grad for backward pass coverage
        torch::Tensor input_grad = input.clone().detach().requires_grad_(true);
        torch::Tensor target_no_grad = target.clone().detach();
        torch::Tensor variance_no_grad = variance.clone().detach();
        
        try {
            // For backward pass, use mean reduction to get a scalar
            auto grad_options = torch::nn::GaussianNLLLossOptions()
                .full(full)
                .eps(eps)
                .reduction(reduction == torch::kNone ? torch::kMean : reduction);
            torch::nn::GaussianNLLLoss loss_fn_grad(grad_options);
            
            torch::Tensor loss_grad = loss_fn_grad(input_grad, target_no_grad, variance_no_grad);
            
            if (loss_grad.defined() && loss_grad.numel() == 1) {
                loss_grad.backward();
            }
        } catch (...) {
            // Backward pass failed silently - expected for some configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}