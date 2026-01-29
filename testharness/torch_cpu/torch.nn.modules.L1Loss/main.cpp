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
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has elements
        if (input.numel() == 0) {
            return 0;
        }
        
        // Create target tensor with the SAME shape as input
        // This is required for L1Loss
        torch::Tensor target;
        if (offset < Size) {
            // Create a tensor from remaining data and reshape to match input
            torch::Tensor raw_target = fuzzer_utils::createTensor(Data, Size, offset);
            // Use input's shape to create target
            target = torch::randn(input.sizes(), input.options());
            // Fill with values derived from raw_target if possible
            if (raw_target.numel() > 0) {
                float scale = raw_target.flatten()[0].item<float>();
                if (std::isfinite(scale)) {
                    target = target * scale;
                }
            }
        } else {
            // Clone input and modify to create target
            target = input.clone() + 1.0f;
        }
        
        // Determine reduction mode from fuzzer data
        uint8_t reduction_selector = 0;
        if (offset < Size) {
            reduction_selector = Data[offset++];
        }
        
        // Test with the selected reduction mode
        switch (reduction_selector % 3) {
            case 0: {
                torch::nn::L1Loss l1_loss(torch::nn::L1LossOptions().reduction(torch::kNone));
                torch::Tensor loss = l1_loss(input, target);
                break;
            }
            case 1: {
                torch::nn::L1Loss l1_loss(torch::nn::L1LossOptions().reduction(torch::kMean));
                torch::Tensor loss = l1_loss(input, target);
                break;
            }
            case 2: {
                torch::nn::L1Loss l1_loss(torch::nn::L1LossOptions().reduction(torch::kSum));
                torch::Tensor loss = l1_loss(input, target);
                break;
            }
        }
        
        // Test all reduction modes to maximize coverage
        {
            torch::nn::L1Loss l1_loss_none(torch::nn::L1LossOptions().reduction(torch::kNone));
            torch::Tensor loss_none = l1_loss_none(input, target);
        }
        
        {
            torch::nn::L1Loss l1_loss_mean(torch::nn::L1LossOptions().reduction(torch::kMean));
            torch::Tensor loss_mean = l1_loss_mean(input, target);
        }
        
        {
            torch::nn::L1Loss l1_loss_sum(torch::nn::L1LossOptions().reduction(torch::kSum));
            torch::Tensor loss_sum = l1_loss_sum(input, target);
        }
        
        // Test with different dtypes
        if (offset < Size && input.is_floating_point()) {
            try {
                torch::Tensor input_double = input.to(torch::kDouble);
                torch::Tensor target_double = target.to(torch::kDouble);
                torch::nn::L1Loss l1_loss_double(torch::nn::L1LossOptions().reduction(torch::kMean));
                torch::Tensor loss_double = l1_loss_double(input_double, target_double);
            } catch (...) {
                // Silently ignore dtype conversion issues
            }
        }
        
        // Test with requires_grad for backward compatibility
        if (offset < Size) {
            try {
                torch::Tensor input_grad = input.clone().detach().requires_grad_(true);
                torch::Tensor target_no_grad = target.clone().detach();
                
                torch::nn::L1Loss l1_loss_grad(torch::nn::L1LossOptions().reduction(torch::kMean));
                torch::Tensor loss_grad = l1_loss_grad(input_grad, target_no_grad);
                
                // Test backward pass
                loss_grad.backward();
            } catch (...) {
                // Silently ignore gradient computation issues
            }
        }
        
        // Test with batched inputs (multi-dimensional)
        if (offset + 2 < Size) {
            try {
                int64_t batch_size = (Data[offset] % 4) + 1;
                int64_t features = (Data[offset + 1] % 8) + 1;
                offset += 2;
                
                torch::Tensor batched_input = torch::randn({batch_size, features});
                torch::Tensor batched_target = torch::randn({batch_size, features});
                
                torch::nn::L1Loss l1_loss_batched(torch::nn::L1LossOptions().reduction(torch::kMean));
                torch::Tensor loss_batched = l1_loss_batched(batched_input, batched_target);
            } catch (...) {
                // Silently ignore shape issues
            }
        }
        
        // Test with extreme values
        if (offset < Size) {
            try {
                torch::Tensor extreme_input = torch::full_like(input, 1e10f);
                torch::Tensor extreme_target = torch::full_like(target, -1e10f);
                
                torch::nn::L1Loss l1_loss_extreme(torch::nn::L1LossOptions().reduction(torch::kMean));
                torch::Tensor extreme_loss = l1_loss_extreme(extreme_input, extreme_target);
            } catch (...) {
                // Silently ignore overflow issues
            }
        }
        
        // Test with zero tensors
        if (offset < Size) {
            try {
                torch::Tensor zero_input = torch::zeros_like(input);
                torch::Tensor zero_target = torch::zeros_like(target);
                
                torch::nn::L1Loss l1_loss_zero(torch::nn::L1LossOptions().reduction(torch::kMean));
                torch::Tensor zero_loss = l1_loss_zero(zero_input, zero_target);
            } catch (...) {
                // Silently ignore
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