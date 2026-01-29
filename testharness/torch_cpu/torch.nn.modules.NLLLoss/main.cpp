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
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract configuration bytes first
        uint8_t batch_byte = Data[offset++];
        uint8_t classes_byte = Data[offset++];
        uint8_t reduction_byte = Data[offset++];
        uint8_t weight_byte = Data[offset++];
        uint8_t ignore_byte = Data[offset++];
        uint8_t use_weight_byte = Data[offset++];
        
        // Derive dimensions
        int64_t batch_size = (batch_byte % 16) + 1;  // 1-16
        int64_t num_classes = (classes_byte % 10) + 2; // 2-11
        
        // Create input tensor (N, C) - log probabilities
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape and prepare input for NLLLoss
        // NLLLoss expects input of shape (N, C) with log probabilities
        int64_t total_elements = input.numel();
        if (total_elements < batch_size * num_classes) {
            // Expand if needed
            input = input.flatten();
            input = torch::cat({input, torch::zeros(batch_size * num_classes - total_elements)});
        }
        input = input.flatten().slice(0, 0, batch_size * num_classes);
        input = input.reshape({batch_size, num_classes}).to(torch::kFloat);
        input.requires_grad_(true);
        
        // Apply log_softmax to get valid log probabilities
        input = torch::log_softmax(input, /*dim=*/1);
        
        // Create target tensor (class indices) of shape (N,)
        torch::Tensor target = torch::zeros({batch_size}, torch::kLong);
        if (offset < Size) {
            // Use remaining data to generate target indices
            for (int64_t i = 0; i < batch_size && offset < Size; i++) {
                target[i] = static_cast<int64_t>(Data[offset++]) % num_classes;
            }
        }
        
        // Determine reduction type
        torch::nn::NLLLossOptions::reduction_t reduction;
        switch (reduction_byte % 3) {
            case 0: reduction = torch::kMean; break;
            case 1: reduction = torch::kSum; break;
            case 2: reduction = torch::kNone; break;
        }
        
        // Create options
        torch::nn::NLLLossOptions options;
        options.reduction(reduction);
        
        // Optionally set weight
        if (use_weight_byte % 2 == 1) {
            float weight_scale = static_cast<float>(weight_byte) / 255.0f * 2.0f + 0.1f;
            torch::Tensor weight = torch::ones({num_classes}, torch::kFloat) * weight_scale;
            options.weight(weight);
        }
        
        // Set ignore_index (may or may not match any target)
        int64_t ignore_index = (static_cast<int64_t>(ignore_byte) % (num_classes + 10)) - 5;
        options.ignore_index(ignore_index);
        
        // Create NLLLoss module
        torch::nn::NLLLoss nll_loss(options);
        
        // Forward pass
        torch::Tensor output = nll_loss->forward(input, target);
        
        // Backward pass
        try {
            if (output.requires_grad() && output.numel() == 1) {
                output.backward();
            } else if (output.requires_grad() && output.numel() > 1) {
                // For reduction=none, output has shape (N,)
                output.sum().backward();
            }
        } catch (...) {
            // Ignore backward errors silently
        }
        
        // Also test functional interface
        try {
            torch::Tensor input2 = torch::randn({batch_size, num_classes}, torch::kFloat);
            input2 = torch::log_softmax(input2, 1);
            torch::Tensor output2 = torch::nn::functional::nll_loss(
                input2, target,
                torch::nn::functional::NLLLossFuncOptions().reduction(reduction)
            );
        } catch (...) {
            // Ignore functional API errors silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}