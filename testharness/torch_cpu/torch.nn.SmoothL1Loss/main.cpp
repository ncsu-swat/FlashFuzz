#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor - try to make it same shape as input for valid loss computation
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            // Reshape target to match input shape if possible
            try {
                if (target.numel() != input.numel()) {
                    target = torch::randn_like(input);
                } else if (!target.sizes().equals(input.sizes())) {
                    target = target.reshape(input.sizes());
                }
            } catch (...) {
                target = torch::randn_like(input);
            }
        } else {
            target = torch::randn_like(input);
        }
        
        // Get reduction mode from remaining data
        int reduction_selector = 1; // default to kMean
        if (offset < Size) {
            reduction_selector = Data[offset++] % 3;
        }
        
        // Get beta value from remaining data
        double beta = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Check for NaN or Inf
            if (std::isnan(beta) || std::isinf(beta)) {
                beta = 1.0;
            }
            
            // Ensure beta is positive (as required by the API)
            beta = std::abs(beta);
            
            // Clamp to reasonable range
            if (beta > 1e10) {
                beta = 1e10;
            }
            if (beta < 1e-10 && beta != 0.0) {
                beta = 1e-10;
            }
        }
        
        // Create SmoothL1Loss module with the options
        torch::nn::SmoothL1LossOptions options;
        options.beta(beta);
        
        if (reduction_selector == 0) {
            options.reduction(torch::kNone);
        } else if (reduction_selector == 1) {
            options.reduction(torch::kMean);
        } else {
            options.reduction(torch::kSum);
        }
        
        torch::nn::SmoothL1Loss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor loss = loss_fn(input, target);
        
        // Try backward pass if tensors require grad
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                auto input_grad = input.clone().detach().set_requires_grad(true);
                auto target_detached = target.clone().detach();
                
                auto loss_with_grad = loss_fn(input_grad, target_detached);
                
                if (loss_with_grad.numel() > 0) {
                    if (reduction_selector == 0) {
                        // For 'none' reduction, loss has same shape as input
                        auto grad_output = torch::ones_like(loss_with_grad);
                        loss_with_grad.backward(grad_output);
                    } else {
                        // For 'mean' or 'sum' reduction, the loss is a scalar
                        loss_with_grad.backward();
                    }
                }
            } catch (...) {
                // Backward pass may fail for various reasons, silently ignore
            }
        }
        
        // Test with different tensor types
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                auto input_float = input.to(torch::kFloat32);
                auto target_float = target.to(torch::kFloat32);
                loss_fn(input_float, target_float);
            } catch (...) {
                // Type conversion may fail, silently ignore
            }
        }
        
        // Test with double precision
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                auto input_double = input.to(torch::kFloat64);
                auto target_double = target.to(torch::kFloat64);
                loss_fn(input_double, target_double);
            } catch (...) {
                // Type conversion may fail, silently ignore
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