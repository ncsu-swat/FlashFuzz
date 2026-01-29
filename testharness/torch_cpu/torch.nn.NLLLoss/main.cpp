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
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract configuration from data
        uint8_t batch_size_byte = Data[offset++];
        uint8_t num_classes_byte = Data[offset++];
        uint8_t reduction_byte = Data[offset++];
        uint8_t use_weight_byte = Data[offset++];
        uint8_t ignore_idx_byte = Data[offset++];
        
        // Derive dimensions (ensure valid ranges)
        int64_t batch_size = (batch_size_byte % 16) + 1;  // 1-16
        int64_t num_classes = (num_classes_byte % 10) + 2; // 2-11 classes
        
        // Create input tensor and apply log_softmax (NLLLoss expects log-probabilities)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape input to (N, C) format for NLLLoss
        try {
            input = input.reshape({batch_size, num_classes}).to(torch::kFloat);
        } catch (...) {
            // If reshape fails, create a proper tensor
            input = torch::randn({batch_size, num_classes});
        }
        
        // Apply log_softmax to get proper log-probabilities
        input = torch::log_softmax(input, /*dim=*/1);
        
        // Create target tensor with class indices (shape: N)
        torch::Tensor target;
        if (offset + batch_size <= Size) {
            std::vector<int64_t> target_vals;
            for (int64_t i = 0; i < batch_size && offset < Size; i++) {
                target_vals.push_back(Data[offset++] % num_classes);
            }
            target = torch::tensor(target_vals, torch::kLong);
        } else {
            target = torch::randint(0, num_classes, {batch_size}, torch::kLong);
        }
        
        // Ensure target has correct size
        if (target.size(0) != batch_size) {
            target = torch::randint(0, num_classes, {batch_size}, torch::kLong);
        }
        
        // Reduction type: 0=None, 1=Mean, 2=Sum
        int reduction_type = reduction_byte % 3;
        
        // Configure ignore_index
        int64_t ignore_index = -100;
        if (ignore_idx_byte % 4 == 0) {
            // Sometimes use a valid class index to ignore
            ignore_index = ignore_idx_byte % num_classes;
        }
        
        // Create weight tensor (optional)
        torch::Tensor weight;
        bool use_weight = (use_weight_byte % 2 == 0);
        
        if (use_weight) {
            weight = torch::ones({num_classes});
            for (int64_t i = 0; i < num_classes && offset < Size; i++) {
                weight[i] = static_cast<float>(Data[offset++]) / 255.0f + 0.1f;
            }
        }
        
        // Helper lambda to create options with the right reduction
        auto make_options = [&](bool with_weight) {
            torch::nn::NLLLossOptions opts;
            opts.ignore_index(ignore_index);
            if (with_weight) {
                opts.weight(weight);
            }
            switch (reduction_type) {
                case 0: opts.reduction(torch::kNone); break;
                case 1: opts.reduction(torch::kMean); break;
                default: opts.reduction(torch::kSum); break;
            }
            return opts;
        };
        
        // Test NLLLoss module
        {
            torch::nn::NLLLoss nll_loss(make_options(use_weight));
            
            torch::Tensor output = nll_loss->forward(input, target);
            
            // Verify output shape based on reduction
            if (reduction_type == 0) {
                // Output should have shape (N)
                (void)output.size(0);
            } else {
                // Output should be scalar
                (void)output.item<float>();
            }
        }
        
        // Test functional version
        {
            torch::nn::functional::NLLLossFuncOptions func_opts;
            func_opts.ignore_index(ignore_index);
            if (use_weight) {
                func_opts.weight(weight);
            }
            switch (reduction_type) {
                case 0: func_opts.reduction(torch::kNone); break;
                case 1: func_opts.reduction(torch::kMean); break;
                default: func_opts.reduction(torch::kSum); break;
            }
            
            torch::Tensor output_functional = torch::nn::functional::nll_loss(
                input, target, func_opts);
        }
        
        // Test with 2D NLLLoss (for image segmentation style input)
        if (offset + 4 < Size) {
            int64_t height = (Data[offset++] % 4) + 2;  // 2-5
            int64_t width = (Data[offset++] % 4) + 2;   // 2-5
            
            try {
                torch::Tensor input_2d = torch::randn({batch_size, num_classes, height, width});
                input_2d = torch::log_softmax(input_2d, /*dim=*/1);
                
                torch::Tensor target_2d = torch::randint(0, num_classes, {batch_size, height, width}, torch::kLong);
                
                torch::nn::NLLLoss nll_loss_2d(make_options(false));
                
                torch::Tensor output_2d = nll_loss_2d->forward(input_2d, target_2d);
            } catch (...) {
                // Silently handle shape mismatches in 2D case
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