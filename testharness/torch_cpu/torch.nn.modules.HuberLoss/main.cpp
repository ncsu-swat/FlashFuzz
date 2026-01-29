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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure both tensors are float type for loss computation
        input = input.to(torch::kFloat32);
        target = target.to(torch::kFloat32);
        
        // Extract delta parameter from the input data
        double delta = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&delta, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure delta is positive and finite
            if (!std::isfinite(delta) || delta <= 0.0) {
                delta = 1.0;
            }
        }
        
        // Extract reduction parameter from the input data
        uint8_t reduction_byte = 0;
        if (offset < Size) {
            reduction_byte = Data[offset++];
        }
        
        // Map reduction_byte to one of the three reduction modes
        torch::nn::HuberLossOptions::reduction_t reduction;
        switch (reduction_byte % 3) {
            case 0:
                reduction = torch::kNone;
                break;
            case 1:
                reduction = torch::kMean;
                break;
            case 2:
                reduction = torch::kSum;
                break;
            default:
                reduction = torch::kMean;
        }
        
        // Create HuberLoss module with the extracted parameters
        torch::nn::HuberLossOptions options;
        options.delta(delta);
        options.reduction(reduction);
        torch::nn::HuberLoss huber_loss(options);
        
        // Apply the HuberLoss operation
        // This may throw if input and target have incompatible shapes
        torch::Tensor loss = huber_loss->forward(input, target);
        
        // Ensure the loss is computed (force evaluation)
        if (loss.defined()) {
            // Use sum() to get a scalar regardless of reduction mode
            auto loss_sum = loss.sum().item<double>();
            (void)loss_sum;
        }
        
        // Try with different delta values (silent catch for expected failures)
        try {
            if (offset + sizeof(double) <= Size) {
                double new_delta;
                std::memcpy(&new_delta, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                // Sanitize new_delta
                if (!std::isfinite(new_delta)) {
                    new_delta = 1.0;
                }
                
                // Try with a very small delta
                double small_delta = std::abs(new_delta) * 1e-5 + 1e-10;
                torch::nn::HuberLossOptions small_options;
                small_options.delta(small_delta);
                small_options.reduction(reduction);
                torch::nn::HuberLoss small_huber_loss(small_options);
                torch::Tensor small_loss = small_huber_loss->forward(input, target);
                (void)small_loss.sum().item<double>();
                
                // Try with a very large delta
                double large_delta = std::abs(new_delta) * 1e5 + 1.0;
                if (large_delta > 1e10) large_delta = 1e10; // Cap to avoid overflow
                torch::nn::HuberLossOptions large_options;
                large_options.delta(large_delta);
                large_options.reduction(reduction);
                torch::nn::HuberLoss large_huber_loss(large_options);
                torch::Tensor large_loss = large_huber_loss->forward(input, target);
                (void)large_loss.sum().item<double>();
            }
        } catch (...) {
            // Silently ignore - these are expected failures
        }
        
        // Try with different reduction modes (silent catch for expected failures)
        try {
            torch::nn::HuberLossOptions none_options;
            none_options.delta(delta);
            none_options.reduction(torch::kNone);
            torch::nn::HuberLoss none_huber_loss(none_options);
            torch::Tensor none_loss = none_huber_loss->forward(input, target);
            (void)none_loss.sum().item<double>();
            
            torch::nn::HuberLossOptions sum_options;
            sum_options.delta(delta);
            sum_options.reduction(torch::kSum);
            torch::nn::HuberLoss sum_huber_loss(sum_options);
            torch::Tensor sum_loss = sum_huber_loss->forward(input, target);
            (void)sum_loss.item<double>();
            
            torch::nn::HuberLossOptions mean_options;
            mean_options.delta(delta);
            mean_options.reduction(torch::kMean);
            torch::nn::HuberLoss mean_huber_loss(mean_options);
            torch::Tensor mean_loss = mean_huber_loss->forward(input, target);
            (void)mean_loss.item<double>();
        } catch (...) {
            // Silently ignore - these are expected failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}