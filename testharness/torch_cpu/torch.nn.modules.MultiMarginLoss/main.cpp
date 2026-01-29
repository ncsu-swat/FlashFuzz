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
        if (Size < 8) {
            return 0;
        }
        
        // Extract batch size and num_classes from fuzzer data
        int64_t batch_size = (Data[offset++] % 16) + 1;  // 1-16
        int64_t num_classes = (Data[offset++] % 10) + 2; // 2-11 classes
        
        // Create input tensor of shape (N, C) - scores for each class
        torch::Tensor input = torch::randn({batch_size, num_classes}, 
            torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
        
        // Modify input based on fuzzer data
        if (offset + 4 <= Size) {
            float scale = static_cast<float>(Data[offset++]) / 25.5f + 0.1f;
            input = input * scale;
        }
        
        // Create target tensor of shape (N) with class indices in [0, C-1]
        torch::Tensor target = torch::zeros({batch_size}, torch::kLong);
        for (int64_t i = 0; i < batch_size && offset < Size; i++) {
            target[i] = static_cast<int64_t>(Data[offset++] % num_classes);
        }
        
        // Extract parameters for MultiMarginLoss
        int64_t p = 1;
        double margin = 1.0;
        torch::Tensor weight;
        torch::nn::MultiMarginLossOptions::reduction_t reduction = torch::kMean;
        
        // Extract p value (1 or 2)
        if (offset < Size) {
            p = (Data[offset++] % 2) + 1;
        }
        
        // Extract margin value
        if (offset + 2 <= Size) {
            uint16_t margin_bits;
            std::memcpy(&margin_bits, Data + offset, sizeof(margin_bits));
            offset += sizeof(margin_bits);
            margin = static_cast<double>(margin_bits) / 1000.0 + 0.1;
        }
        
        // Create weight tensor if indicated by fuzzer data
        if (offset < Size && Data[offset++] % 2 == 1) {
            // Weight must be 1D tensor of size num_classes
            weight = torch::rand({num_classes}, torch::kFloat32) + 0.1f;
        }
        
        // Extract reduction mode
        if (offset < Size) {
            int reduction_mode = Data[offset++] % 3;
            if (reduction_mode == 0) {
                reduction = torch::kNone;
            } else if (reduction_mode == 1) {
                reduction = torch::kMean;
            } else {
                reduction = torch::kSum;
            }
        }
        
        // Create MultiMarginLoss module with options
        torch::nn::MultiMarginLossOptions options;
        options.p(p).margin(margin).reduction(reduction);
        
        if (weight.defined()) {
            options.weight(weight);
        }
        
        auto loss_fn = torch::nn::MultiMarginLoss(options);
        
        // Apply the loss function
        torch::Tensor output;
        try {
            output = loss_fn->forward(input, target);
        } catch (const std::exception &) {
            // Shape mismatches or invalid configurations are expected
            return 0;
        }
        
        // Compute gradients if output is scalar (reduction != None)
        if (offset < Size && Data[offset] % 2 == 0) {
            if (output.defined() && output.numel() == 1) {
                try {
                    output.backward();
                } catch (const std::exception &) {
                    // Gradient computation may fail, that's okay
                }
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