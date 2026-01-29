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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor with gradients enabled
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor variance = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure all tensors are floating point
        if (!torch::isFloatingType(input.scalar_type())) {
            input = input.to(torch::kFloat);
        }
        if (!torch::isFloatingType(variance.scalar_type())) {
            variance = variance.to(torch::kFloat);
        }
        if (!torch::isFloatingType(target.scalar_type())) {
            target = target.to(torch::kFloat);
        }
        
        // Ensure all tensors have the same dtype
        variance = variance.to(input.scalar_type());
        target = target.to(input.scalar_type());
        
        // Ensure variance is positive (required by GaussianNLLLoss)
        variance = torch::abs(variance) + 1e-6;
        
        // Make shapes compatible: use input's shape for all tensors
        // GaussianNLLLoss requires input, target to have same shape
        // variance can be same shape or broadcastable
        if (input.numel() > 0) {
            auto input_sizes = input.sizes().vec();
            
            // Reshape target to match input
            int64_t input_numel = input.numel();
            target = target.flatten();
            if (target.numel() < input_numel) {
                // Expand by repeating
                int64_t repeats = (input_numel / target.numel()) + 1;
                target = target.repeat({repeats});
            }
            target = target.narrow(0, 0, input_numel).reshape(input_sizes);
            
            // Reshape variance to match input
            variance = variance.flatten();
            if (variance.numel() < input_numel) {
                int64_t repeats = (input_numel / variance.numel()) + 1;
                variance = variance.repeat({repeats});
            }
            variance = variance.narrow(0, 0, input_numel).reshape(input_sizes);
            // Re-ensure variance is positive after reshape
            variance = torch::abs(variance) + 1e-6;
        }
        
        // Enable gradients for backward pass
        input = input.clone().detach().requires_grad_(true);
        
        // Parse reduction mode from the remaining data (0=None, 1=Mean, 2=Sum)
        int reduction_mode = 1; // Default to Mean
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            reduction_mode = reduction_byte % 3;
        }
        
        // Parse full parameter
        bool full = false;
        if (offset < Size) {
            full = Data[offset++] & 1;
        }
        
        // Parse eps parameter
        double eps = 1e-6;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure eps is positive and reasonable
            eps = std::abs(eps);
            if (std::isnan(eps) || std::isinf(eps)) {
                eps = 1e-6;
            }
            if (eps < 1e-12) eps = 1e-12;
            if (eps > 1.0) eps = 1.0;
        }
        
        // Create GaussianNLLLoss module with appropriate options
        torch::nn::GaussianNLLLossOptions options;
        options.full(full);
        options.eps(eps);
        
        // Set reduction mode
        if (reduction_mode == 0) {
            options.reduction(torch::kNone);
        } else if (reduction_mode == 1) {
            options.reduction(torch::kMean);
        } else {
            options.reduction(torch::kSum);
        }
        
        torch::nn::GaussianNLLLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor loss = loss_fn(input, target, variance);
        
        // Perform backward pass if possible
        if (loss.numel() > 0 && loss.requires_grad()) {
            try {
                // For reduction=None, we need to sum before backward
                if (reduction_mode == 0) {
                    loss.sum().backward();
                } else {
                    loss.backward();
                }
            } catch (...) {
                // Backward pass failed, but we can continue
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}