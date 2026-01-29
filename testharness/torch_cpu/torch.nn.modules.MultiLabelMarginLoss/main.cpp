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
        
        // Need at least some data to create tensors and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract batch size and number of classes from fuzzer data
        uint8_t batch_size = (Data[offset++] % 8) + 1;  // 1-8
        uint8_t num_classes = (Data[offset++] % 16) + 2; // 2-17
        
        // Create input tensor of shape [batch_size, num_classes]
        torch::Tensor input = torch::randn({batch_size, num_classes}, 
            torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
        
        // Initialize input with fuzzer data if available
        if (offset + batch_size * num_classes <= Size) {
            auto input_accessor = input.accessor<float, 2>();
            for (int i = 0; i < batch_size; i++) {
                for (int j = 0; j < num_classes; j++) {
                    if (offset < Size) {
                        input_accessor[i][j] = static_cast<float>(Data[offset++]) / 25.5f - 5.0f;
                    }
                }
            }
        }
        
        // Create target tensor of shape [batch_size, num_classes]
        // Target contains class indices (0 to C-1) and -1 for padding
        torch::Tensor target = torch::full({batch_size, num_classes}, -1, torch::kLong);
        
        // Fill target with valid class indices based on fuzzer data
        if (offset < Size) {
            auto target_accessor = target.accessor<int64_t, 2>();
            for (int i = 0; i < batch_size && offset < Size; i++) {
                // Number of positive labels for this sample
                uint8_t num_labels = Data[offset++] % num_classes;
                for (int j = 0; j < num_labels && j < num_classes && offset < Size; j++) {
                    int64_t class_idx = Data[offset++] % num_classes;
                    target_accessor[i][j] = class_idx;
                }
            }
        }
        
        // Determine reduction mode
        torch::nn::MultiLabelMarginLossOptions::reduction_t reduction;
        if (offset < Size) {
            uint8_t reduction_val = Data[offset++] % 3;
            if (reduction_val == 0) {
                reduction = torch::kNone;
            } else if (reduction_val == 1) {
                reduction = torch::kMean;
            } else {
                reduction = torch::kSum;
            }
        } else {
            reduction = torch::kMean;
        }
        
        // Create the loss module
        torch::nn::MultiLabelMarginLoss loss_module(
            torch::nn::MultiLabelMarginLossOptions().reduction(reduction)
        );
        
        // Compute the loss
        torch::Tensor output = loss_module->forward(input, target);
        
        // Perform backward pass
        try {
            if (output.requires_grad() || (output.dim() == 0 && input.requires_grad())) {
                if (output.dim() == 0) {
                    output.backward();
                } else {
                    // For reduction=None, output has shape [batch_size]
                    output.sum().backward();
                }
            }
        } catch (...) {
            // Backward pass failures are expected in some cases
        }
        
        // Test with 1D input (single sample case)
        if (offset + 4 < Size) {
            try {
                uint8_t single_classes = (Data[offset++] % 16) + 2;
                torch::Tensor input_1d = torch::randn({single_classes}, 
                    torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
                
                // Create 1D target with class indices
                torch::Tensor target_1d = torch::full({single_classes}, -1, torch::kLong);
                uint8_t num_labels = Data[offset++] % single_classes;
                auto target_1d_accessor = target_1d.accessor<int64_t, 1>();
                for (int j = 0; j < num_labels && offset < Size; j++) {
                    target_1d_accessor[j] = Data[offset++] % single_classes;
                }
                
                torch::Tensor output_1d = loss_module->forward(input_1d, target_1d);
            } catch (...) {
                // Shape mismatches expected
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