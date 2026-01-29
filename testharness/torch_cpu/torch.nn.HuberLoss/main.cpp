#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isnan, std::isinf, std::abs

// --- Fuzzer Entry Point ---
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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor with the same shape as input for valid loss computation
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to make shapes compatible - reshape target to match input if possible
        try {
            if (target.numel() != input.numel()) {
                // If element counts differ, create a new target with same shape
                target = torch::randn_like(input);
            } else if (!target.sizes().equals(input.sizes())) {
                // Same number of elements but different shape - reshape
                target = target.reshape(input.sizes());
            }
        } catch (...) {
            // If reshaping fails, create a random target with matching shape
            target = torch::randn_like(input);
        }
        
        // Parse reduction mode from input data
        torch::nn::HuberLossOptions::reduction_t reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0:
                    reduction_mode = torch::kNone;
                    break;
                case 1:
                    reduction_mode = torch::kMean;
                    break;
                case 2:
                    reduction_mode = torch::kSum;
                    break;
            }
        }
        
        // Parse delta value from input data
        double delta = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&delta, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure delta is valid (positive and finite)
            if (std::isnan(delta) || std::isinf(delta)) {
                delta = 1.0;
            } else if (delta <= 0) {
                delta = std::abs(delta);
                if (delta == 0 || delta < 1e-6) {
                    delta = 1.0;
                }
            }
            // Clamp to reasonable range
            if (delta > 1e6) {
                delta = 1e6;
            }
        }
        
        // Create HuberLoss module with options
        torch::nn::HuberLossOptions options;
        options.reduction(reduction_mode).delta(delta);
        torch::nn::HuberLoss huber_loss(options);
        
        // Apply HuberLoss
        torch::Tensor loss = huber_loss(input, target);
        
        // Ensure computation is performed by accessing the result
        volatile float loss_value = 0.0f;
        if (loss.numel() > 0) {
            if (loss.dim() == 0) {
                loss_value = loss.item<float>();
            } else {
                loss_value = loss.sum().item<float>();
            }
        }
        (void)loss_value; // Suppress unused variable warning
        
        // Test with different delta values
        if (offset + sizeof(double) <= Size) {
            double delta2 = 0.5;
            std::memcpy(&delta2, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure delta is valid (positive and finite)
            if (std::isnan(delta2) || std::isinf(delta2)) {
                delta2 = 0.5;
            } else if (delta2 <= 0) {
                delta2 = std::abs(delta2);
                if (delta2 == 0 || delta2 < 1e-6) {
                    delta2 = 0.5;
                }
            }
            // Clamp to reasonable range
            if (delta2 > 1e6) {
                delta2 = 1e6;
            }
            
            torch::nn::HuberLossOptions options2;
            options2.reduction(reduction_mode).delta(delta2);
            torch::nn::HuberLoss huber_loss2(options2);
            
            torch::Tensor loss2 = huber_loss2(input, target);
            
            // Access the result
            volatile float loss_value2 = 0.0f;
            if (loss2.numel() > 0) {
                if (loss2.dim() == 0) {
                    loss_value2 = loss2.item<float>();
                } else {
                    loss_value2 = loss2.sum().item<float>();
                }
            }
            (void)loss_value2; // Suppress unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}