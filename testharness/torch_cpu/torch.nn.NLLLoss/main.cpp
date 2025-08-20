#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor (log probabilities)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (class indices)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure target has integer type for class indices
            if (target.scalar_type() != torch::kLong) {
                target = target.to(torch::kLong);
            }
            
            // Clamp target values to be within valid range for the input's classes
            if (input.dim() > 0 && input.size(1) > 0) {
                int64_t num_classes = input.size(1);
                target = torch::clamp(target, 0, num_classes - 1);
            }
        } else {
            // If we don't have enough data for a target tensor, create a simple one
            if (input.dim() >= 2) {
                int64_t batch_size = input.size(0);
                int64_t num_classes = input.size(1);
                target = torch::randint(0, num_classes, {batch_size}, torch::kLong);
            } else {
                // Default target for edge cases
                target = torch::zeros({1}, torch::kLong);
            }
        }
        
        // Extract parameters for NLLLoss from remaining data
        double weight_val = 1.0;
        int64_t ignore_index = -100;
        torch::enumtype::Reduction reduction = torch::kMean;
        
        if (offset + 1 < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0: reduction = torch::kNone; break;
                case 1: reduction = torch::kMean; break;
                case 2: reduction = torch::kSum; break;
            }
        }
        
        if (offset + 1 < Size) {
            ignore_index = static_cast<int64_t>(Data[offset++]);
        }
        
        // Create weight tensor (optional)
        torch::Tensor weight;
        bool use_weight = false;
        
        if (offset < Size) {
            use_weight = (Data[offset++] % 2 == 0);
            if (use_weight && input.dim() > 1) {
                int64_t num_classes = input.size(1);
                weight = torch::ones({num_classes});
                
                // Fill weight with some values from remaining data
                for (int64_t i = 0; i < num_classes && offset < Size; i++) {
                    weight[i] = static_cast<float>(Data[offset++]) / 255.0f + 0.1f;
                }
            }
        }
        
        // Create NLLLoss module with different configurations
        torch::nn::NLLLoss nll_loss;
        
        if (use_weight) {
            nll_loss = torch::nn::NLLLoss(torch::nn::NLLLossOptions()
                                         .weight(weight)
                                         .ignore_index(ignore_index)
                                         .reduction(reduction));
        } else {
            nll_loss = torch::nn::NLLLoss(torch::nn::NLLLossOptions()
                                         .ignore_index(ignore_index)
                                         .reduction(reduction));
        }
        
        // Apply NLLLoss
        torch::Tensor output = nll_loss->forward(input, target);
        
        // Try functional version as well
        torch::Tensor output_functional;
        if (use_weight) {
            output_functional = torch::nn::functional::nll_loss(
                input, target, 
                torch::nn::functional::NLLLossFuncOptions()
                    .weight(weight)
                    .ignore_index(ignore_index)
                    .reduction(reduction));
        } else {
            output_functional = torch::nn::functional::nll_loss(
                input, target,
                torch::nn::functional::NLLLossFuncOptions()
                    .ignore_index(ignore_index)
                    .reduction(reduction));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}