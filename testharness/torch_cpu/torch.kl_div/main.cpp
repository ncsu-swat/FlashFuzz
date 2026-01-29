#include "fuzzer_utils.h"
#include <iostream>
#include <ATen/ATen.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Create input tensor (log probabilities typically)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have enough data left for parameters
        if (offset >= Size) {
            return 0;
        }

        // Create target tensor with SAME shape as input (required by kl_div)
        torch::Tensor target = torch::rand_like(input);
        
        // Parse reduction mode from input data (0=none, 1=mean, 2=sum, 3=batchmean)
        int64_t reduction_mode = static_cast<int64_t>(Data[offset++] % 4);
        
        // Parse log_target flag from input data
        bool log_target = false;
        if (offset < Size) {
            log_target = Data[offset++] & 0x1;
        }

        // Convert input to float for kl_div (requires floating point)
        input = input.to(torch::kFloat);
        target = target.to(torch::kFloat);
        
        // Ensure target is positive for valid KL divergence computation
        target = target.abs() + 1e-8;
        
        // If log_target is false, target should be probabilities (positive)
        // If log_target is true, target can be log probabilities
        if (log_target) {
            target = target.log();
        }
        
        // Input should be log probabilities
        input = torch::log_softmax(input, -1);

        // Map reduction mode to at::Reduction enum
        int64_t reduction;
        switch (reduction_mode) {
            case 0:
                reduction = at::Reduction::None;
                break;
            case 1:
                reduction = at::Reduction::Mean;
                break;
            case 2:
                reduction = at::Reduction::Sum;
                break;
            case 3:
                // batchmean - use Mean as approximation for at::Reduction
                reduction = at::Reduction::Mean;
                break;
            default:
                reduction = at::Reduction::None;
        }

        // Inner try-catch for operations that may fail with certain inputs
        try {
            // Apply kl_div operation with all parameters
            torch::Tensor result = torch::kl_div(input, target, reduction, log_target);
            (void)result;
        } catch (...) {
            // Silently ignore expected failures
        }

        try {
            // Try with default parameters
            torch::Tensor result2 = torch::kl_div(input, target);
            (void)result2;
        } catch (...) {
            // Silently ignore expected failures
        }

        try {
            // Try with only reduction specified
            torch::Tensor result3 = torch::kl_div(input, target, reduction);
            (void)result3;
        } catch (...) {
            // Silently ignore expected failures
        }

        // Try functional variant
        try {
            torch::nn::functional::KLDivFuncOptions options;
            if (reduction_mode == 0) {
                options.reduction(torch::kNone);
            } else if (reduction_mode == 1) {
                options.reduction(torch::kMean);
            } else if (reduction_mode == 2) {
                options.reduction(torch::kSum);
            } else {
                options.reduction(torch::kBatchMean);
            }
            options.log_target(log_target);
            torch::Tensor result4 = torch::nn::functional::kl_div(input, target, options);
            (void)result4;
        } catch (...) {
            // Silently ignore expected failures
        }

        // Try with double precision
        try {
            torch::Tensor input_double = input.to(torch::kDouble);
            torch::Tensor target_double = target.to(torch::kDouble);
            torch::Tensor result5 = torch::kl_div(input_double, target_double, reduction, log_target);
            (void)result5;
        } catch (...) {
            // Silently ignore expected failures
        }

        // Test with different shaped tensors (batch dimensions)
        if (offset + 2 < Size) {
            try {
                int batch_size = (Data[offset++] % 8) + 1;
                int feature_size = (Data[offset++] % 16) + 1;
                
                torch::Tensor batched_input = torch::randn({batch_size, feature_size});
                torch::Tensor batched_target = torch::rand({batch_size, feature_size}) + 1e-8;
                
                batched_input = torch::log_softmax(batched_input, -1);
                if (log_target) {
                    batched_target = batched_target.log();
                }
                
                torch::Tensor result6 = torch::kl_div(batched_input, batched_target, reduction, log_target);
                (void)result6;
            } catch (...) {
                // Silently ignore expected failures
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