#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor (predictions/logits)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is at least 2D for multi-label classification
        if (input.dim() < 1) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 2) {
            input = input.unsqueeze(0);
        }
        
        // Create target tensor with same shape as input, containing binary values
        torch::Tensor target = torch::zeros_like(input);
        // Fill with binary values based on remaining data
        if (offset < Size) {
            auto target_accessor = target.flatten();
            int64_t num_elements = target_accessor.numel();
            for (int64_t i = 0; i < num_elements && offset < Size; i++) {
                if (Data[offset++] % 2 == 0) {
                    target.flatten()[i] = 1.0f;
                }
            }
            target = target.view(input.sizes());
        } else {
            // Random binary target
            target = torch::randint(0, 2, input.sizes(), torch::kFloat);
        }
        
        // Parse weight tensor (optional) - must match number of classes (last dim)
        torch::Tensor weight;
        bool use_weight = false;
        if (offset < Size) {
            use_weight = (Data[offset++] % 2 == 0);
            if (use_weight) {
                int64_t num_classes = input.size(-1);
                weight = torch::ones({num_classes});
                // Vary weights based on data
                for (int64_t i = 0; i < num_classes && offset < Size; i++) {
                    weight[i] = static_cast<float>(Data[offset++] % 10 + 1) / 10.0f;
                }
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
        
        // Test with module API
        try {
            torch::nn::MultiLabelSoftMarginLossOptions options;
            options.reduction(reduction_mode);
            
            if (use_weight && weight.defined()) {
                options.weight(weight);
            }
            
            torch::nn::MultiLabelSoftMarginLoss loss_fn(options);
            
            torch::Tensor output = loss_fn(input, target);
            
            if (output.defined()) {
                volatile float sum = output.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error&) {
            // Silently catch shape/type mismatches in module API
        }
        
        // Test with functional API
        try {
            // Use the correct name: MultilabelSoftMarginLossFuncOptions (lowercase 'l')
            torch::nn::functional::MultilabelSoftMarginLossFuncOptions func_options;
            func_options.reduction(reduction_mode);
            if (use_weight && weight.defined()) {
                func_options.weight(weight);
            }
            
            torch::Tensor output = torch::nn::functional::multilabel_soft_margin_loss(
                input, target, func_options);
            
            if (output.defined()) {
                volatile float sum = output.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error&) {
            // Silently catch shape/type mismatches in functional API
        }
        
        // Test with different input types
        try {
            torch::Tensor double_input = input.to(torch::kDouble);
            torch::Tensor double_target = target.to(torch::kDouble);
            
            torch::nn::MultiLabelSoftMarginLossOptions double_options;
            double_options.reduction(torch::kMean);
            torch::nn::MultiLabelSoftMarginLoss double_loss_fn(double_options);
            
            torch::Tensor output = double_loss_fn(double_input, double_target);
            if (output.defined()) {
                volatile double sum = output.sum().item<double>();
                (void)sum;
            }
        } catch (const c10::Error&) {
            // Silently catch type-related issues
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}