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
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract configuration parameters first
        uint8_t batch_size_byte = Data[offset++];
        uint8_t num_classes_byte = Data[offset++];
        uint8_t p_byte = Data[offset++];
        uint8_t margin_byte = Data[offset++];
        uint8_t reduction_byte = Data[offset++];
        uint8_t use_weight_byte = Data[offset++];
        
        // Derive meaningful values
        int64_t batch_size = (batch_size_byte % 16) + 1;  // 1-16
        int64_t num_classes = (num_classes_byte % 10) + 2;  // 2-11 classes
        int64_t p_val = (p_byte % 2) + 1;  // 1 or 2
        double margin_val = static_cast<double>(margin_byte) / 255.0 * 5.0 + 0.1;  // 0.1-5.1
        bool use_weight = (use_weight_byte % 2 == 0);
        
        // Determine reduction mode
        torch::nn::MultiMarginLossOptions::reduction_t reduction_mode;
        switch (reduction_byte % 3) {
            case 0:
                reduction_mode = torch::kNone;
                break;
            case 1:
                reduction_mode = torch::kMean;
                break;
            default:
                reduction_mode = torch::kSum;
                break;
        }
        
        // Create input tensor of shape (N, C) with requires_grad for backward pass
        torch::Tensor input = torch::randn({batch_size, num_classes}, 
            torch::TensorOptions().dtype(torch::kFloat).requires_grad(true));
        
        // Modify input based on fuzzer data if available
        if (offset < Size) {
            torch::Tensor fuzz_input = fuzzer_utils::createTensor(Data, Size, offset);
            if (fuzz_input.numel() > 0) {
                fuzz_input = fuzz_input.to(torch::kFloat).flatten();
                int64_t copy_size = std::min(fuzz_input.numel(), input.numel());
                input.flatten().slice(0, 0, copy_size).copy_(
                    fuzz_input.slice(0, 0, copy_size).detach());
                // Re-enable gradients after modification
                input = input.detach().requires_grad_(true);
            }
        }
        
        // Create target tensor with valid class indices [0, num_classes-1]
        torch::Tensor target = torch::zeros({batch_size}, torch::kLong);
        for (int64_t i = 0; i < batch_size && (offset + i) < Size; i++) {
            target[i] = static_cast<int64_t>(Data[offset + i] % num_classes);
        }
        if (offset < Size) {
            offset += batch_size;
        }
        
        // Create MultiMarginLoss options
        torch::nn::MultiMarginLossOptions options;
        options.p(p_val);
        options.margin(margin_val);
        options.reduction(reduction_mode);
        
        // Create and set weight tensor if enabled
        if (use_weight) {
            torch::Tensor weight = torch::ones({num_classes}, torch::kFloat);
            // Vary weights based on fuzzer data
            for (int64_t i = 0; i < num_classes && (offset + i) < Size; i++) {
                weight[i] = static_cast<float>(Data[offset + i]) / 255.0f * 2.0f + 0.1f;
            }
            options.weight(weight);
        }
        
        // Create the loss module
        torch::nn::MultiMarginLoss loss_fn(options);
        
        // Compute loss
        torch::Tensor output;
        try {
            output = loss_fn(input, target);
        } catch (const c10::Error&) {
            // Shape or value errors are expected for some inputs
            return 0;
        }
        
        // Perform backward pass to test gradient computation
        if (output.numel() > 0) {
            if (output.dim() == 0) {
                // Scalar output (mean or sum reduction)
                output.backward();
            } else {
                // Non-scalar output (no reduction)
                torch::Tensor grad_output = torch::ones_like(output);
                output.backward(grad_output);
            }
            
            // Access gradients to ensure they were computed
            if (input.grad().defined()) {
                volatile float grad_sum = input.grad().sum().item<float>();
                (void)grad_sum;
            }
        }
        
        // Test with different input configurations
        if (offset + 4 < Size) {
            // Test with 2D input requiring gradient
            int64_t new_batch = (Data[offset++] % 8) + 1;
            int64_t new_classes = (Data[offset++] % 8) + 2;
            
            torch::Tensor input2 = torch::randn({new_batch, new_classes},
                torch::TensorOptions().dtype(torch::kFloat).requires_grad(true));
            torch::Tensor target2 = torch::zeros({new_batch}, torch::kLong);
            for (int64_t i = 0; i < new_batch && (offset + i) < Size; i++) {
                target2[i] = static_cast<int64_t>(Data[offset + i] % new_classes);
            }
            
            // Create new loss function without weight (different num_classes)
            torch::nn::MultiMarginLossOptions opts2;
            opts2.p(p_val);
            opts2.margin(margin_val);
            opts2.reduction(torch::kMean);
            torch::nn::MultiMarginLoss loss_fn2(opts2);
            
            try {
                torch::Tensor out2 = loss_fn2(input2, target2);
                out2.backward();
            } catch (const c10::Error&) {
                // Expected for invalid configurations
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}