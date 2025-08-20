#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, create one with the same shape as input
            target = torch::ones_like(input);
        }
        
        // Parse weight tensor (optional)
        torch::Tensor weight;
        bool use_weight = false;
        if (offset < Size) {
            use_weight = (Data[offset++] % 2 == 0);
            if (use_weight && offset < Size) {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
            }
        }
        
        // Parse reduction mode
        torch::nn::MultiLabelSoftMarginLossOptions::reduction_t reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_selector = Data[offset++] % 3;
            switch (reduction_selector) {
                case 0:
                    reduction_mode = torch::kNone;
                    break;
                case 1:
                    reduction_mode = torch::kSum;
                    break;
                case 2:
                default:
                    reduction_mode = torch::kMean;
                    break;
            }
        }
        
        // Create MultiLabelSoftMarginLoss module with options
        torch::nn::MultiLabelSoftMarginLossOptions options;
        options.reduction(reduction_mode);
        
        if (use_weight && weight.defined()) {
            options.weight(weight);
        }
        
        torch::nn::MultiLabelSoftMarginLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor output = loss_fn(input, target);
        
        // Ensure the output is used to prevent optimization
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
        }
        
        // Try with functional API as well
        torch::nn::functional::MultiLabelSoftMarginLossFuncOptions func_options;
        func_options.reduction(reduction_mode);
        if (use_weight && weight.defined()) {
            func_options.weight(weight);
        }
        
        output = torch::nn::functional::multilabel_soft_margin_loss(
            input, target, func_options);
        
        // Ensure the output is used
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}