#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

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
            
            // Clamp target values to valid class indices based on input's last dimension
            if (input.dim() > 0 && input.size(-1) > 0) {
                int64_t num_classes = input.size(-1);
                target = torch::clamp(target, 0, num_classes - 1);
            }
        } else {
            // Create a default target if we don't have enough data
            if (input.dim() >= 2) {
                auto batch_size = input.size(0);
                target = torch::zeros({batch_size}, torch::kLong);
            } else {
                target = torch::zeros({1}, torch::kLong);
            }
        }
        
        // Extract parameters for NLLLoss from remaining data
        bool reduction_mean = true;
        float weight_value = 1.0f;
        int64_t ignore_index = -100;
        
        if (offset + 3 <= Size) {
            // Use a byte to determine reduction type
            uint8_t reduction_byte = Data[offset++];
            reduction_mean = (reduction_byte % 3 != 2); // 0,1 -> true (mean/sum), 2 -> false (none)
            
            // Use a byte to determine weight value
            if (offset < Size) {
                uint8_t weight_byte = Data[offset++];
                weight_value = static_cast<float>(weight_byte) / 255.0f * 10.0f;
            }
            
            // Use a byte to determine ignore_index
            if (offset < Size) {
                uint8_t ignore_byte = Data[offset++];
                ignore_index = static_cast<int64_t>(ignore_byte) - 100;
            }
        }
        
        // Create weight tensor if needed
        torch::Tensor weight;
        if (input.dim() > 0 && input.size(-1) > 0) {
            int64_t num_classes = input.size(-1);
            weight = torch::ones({num_classes}, torch::kFloat) * weight_value;
        } else {
            weight = torch::ones({1}, torch::kFloat) * weight_value;
        }
        
        // Create NLLLoss module with different configurations
        torch::nn::NLLLossOptions options;
        
        // Set reduction
        if (reduction_mean) {
            options = options.reduction(torch::kMean);
        } else {
            options = options.reduction(torch::kNone);
        }
        
        // Set weight and ignore_index
        options = options.weight(weight).ignore_index(ignore_index);
        
        torch::nn::NLLLoss nll_loss(options);
        
        // Apply NLLLoss
        torch::Tensor output = nll_loss->forward(input, target);
        
        // Try backward pass if possible
        if (output.requires_grad()) {
            output.backward();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}