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
        
        // Create target tensor (class indices)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure target has integer type for class indices
            if (target.scalar_type() != torch::kLong) {
                target = target.to(torch::kLong);
            }
            
            // Ensure target has proper shape (should be 1D with same batch size as input)
            if (input.dim() > 0 && target.dim() > 0) {
                int64_t batch_size = input.size(0);
                target = target.reshape({batch_size});
            }
        } else {
            // If we don't have enough data for a second tensor, create a simple target
            if (input.dim() > 0) {
                int64_t batch_size = input.size(0);
                target = torch::zeros({batch_size}, torch::kLong);
            } else {
                target = torch::zeros({1}, torch::kLong);
            }
        }
        
        // Extract configuration parameters from remaining data
        double weight_val = 1.0;
        int64_t p_val = 1;
        double margin_val = 1.0;
        
        if (offset + 2 < Size) {
            // Use some bytes to determine p value (1 or 2)
            p_val = (Data[offset++] % 2) + 1;
            
            // Use some bytes for margin
            uint8_t margin_byte = Data[offset++];
            margin_val = static_cast<double>(margin_byte) / 255.0 * 10.0;
            
            // Use some bytes for weight
            if (offset < Size) {
                uint8_t weight_byte = Data[offset++];
                weight_val = static_cast<double>(weight_byte) / 255.0 * 10.0;
            }
        }
        
        // Create weight tensor (optional)
        torch::Tensor weight;
        bool use_weight = false;
        
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            use_weight = true;
            if (offset < Size) {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure weight has proper type
                if (weight.scalar_type() != torch::kFloat && 
                    weight.scalar_type() != torch::kDouble) {
                    weight = weight.to(torch::kFloat);
                }
            } else {
                // Create a default weight tensor
                if (input.dim() > 1) {
                    int64_t num_classes = input.size(1);
                    weight = torch::ones({num_classes}, torch::kFloat) * weight_val;
                } else {
                    weight = torch::ones({1}, torch::kFloat) * weight_val;
                }
            }
        }
        
        // Create reduction mode
        torch::nn::MultiMarginLossOptions::reduction_t reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            if (reduction_byte % 3 == 0) {
                reduction_mode = torch::kNone;
            } else if (reduction_byte % 3 == 1) {
                reduction_mode = torch::kMean;
            } else {
                reduction_mode = torch::kSum;
            }
        }
        
        // Create MultiMarginLoss module with various configurations
        torch::nn::MultiMarginLossOptions options;
        options.p(p_val).margin(margin_val).reduction(reduction_mode);
        
        if (use_weight) {
            options.weight(weight);
        }
        
        torch::nn::MultiMarginLoss loss_fn(options);
        
        // Convert input to float if needed
        if (input.scalar_type() != torch::kFloat && 
            input.scalar_type() != torch::kDouble) {
            input = input.to(torch::kFloat);
        }
        
        // Apply the loss function
        torch::Tensor output = loss_fn(input, target);
        
        // Perform backward pass to test gradients
        if (output.numel() > 0 && output.requires_grad()) {
            output.backward();
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}