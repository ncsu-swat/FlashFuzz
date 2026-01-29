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
        // Skip if we don't have enough data
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract dimensions from fuzzer data
        uint8_t batch_size_raw = Data[offset++];
        uint8_t num_classes_raw = Data[offset++];
        uint8_t reduction_selector = Data[offset++];
        
        // Ensure reasonable dimensions (1-16 for batch, 2-32 for classes)
        int64_t batch_size = (batch_size_raw % 16) + 1;
        int64_t num_classes = (num_classes_raw % 31) + 2;
        
        // Determine reduction mode
        torch::nn::MultiLabelMarginLossOptions::reduction_t reduction_mode;
        switch (reduction_selector % 3) {
            case 0: reduction_mode = torch::kNone; break;
            case 1: reduction_mode = torch::kMean; break;
            case 2: reduction_mode = torch::kSum; break;
            default: reduction_mode = torch::kMean; break;
        }
        
        // Create input tensor (batch_size x num_classes) with float type
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape and convert input to proper format
        input = input.to(torch::kFloat).reshape({-1});
        int64_t total_elements = batch_size * num_classes;
        
        if (input.numel() < total_elements) {
            // Pad with zeros if not enough elements
            torch::Tensor padding = torch::zeros({total_elements - input.numel()});
            input = torch::cat({input, padding});
        } else if (input.numel() > total_elements) {
            input = input.slice(0, 0, total_elements);
        }
        
        input = input.reshape({batch_size, num_classes});
        input = input.clone().detach().requires_grad_(true);
        
        // Create target tensor with valid class indices
        // Target should be LongTensor with values in [0, C-1] or -1 for padding
        std::vector<int64_t> target_data;
        target_data.reserve(batch_size * num_classes);
        
        for (int64_t b = 0; b < batch_size; b++) {
            // For each sample, assign some valid class labels then pad with -1
            int64_t num_labels = 1;
            if (offset < Size) {
                num_labels = (Data[offset++] % (num_classes - 1)) + 1;
            }
            
            for (int64_t c = 0; c < num_classes; c++) {
                if (c < num_labels) {
                    // Valid class index
                    int64_t class_idx = 0;
                    if (offset < Size) {
                        class_idx = Data[offset++] % num_classes;
                    } else {
                        class_idx = c % num_classes;
                    }
                    target_data.push_back(class_idx);
                } else {
                    // Padding with -1
                    target_data.push_back(-1);
                }
            }
        }
        
        torch::Tensor target = torch::tensor(target_data, torch::kLong).reshape({batch_size, num_classes});
        
        // Create MultiLabelMarginLoss module
        auto options = torch::nn::MultiLabelMarginLossOptions().reduction(reduction_mode);
        auto loss_fn = torch::nn::MultiLabelMarginLoss(options);
        
        // Apply the loss function
        torch::Tensor output;
        try {
            output = loss_fn->forward(input, target);
        } catch (const std::exception&) {
            // Shape mismatch or other expected errors - silently ignore
            return 0;
        }
        
        // Try backward pass if output requires grad
        if (output.numel() > 0 && output.requires_grad()) {
            try {
                if (output.dim() == 0) {
                    output.backward();
                } else {
                    output.sum().backward();
                }
            } catch (const std::exception&) {
                // Ignore backward exceptions
            }
        }
        
        // Also test 1D input variant
        if (batch_size > 0 && offset < Size) {
            try {
                torch::Tensor input_1d = input[0].clone().detach().requires_grad_(true);
                torch::Tensor target_1d = target[0].clone();
                
                torch::Tensor output_1d = loss_fn->forward(input_1d, target_1d);
                if (output_1d.numel() > 0 && output_1d.requires_grad()) {
                    if (output_1d.dim() == 0) {
                        output_1d.backward();
                    }
                }
            } catch (const std::exception&) {
                // Ignore 1D variant exceptions
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