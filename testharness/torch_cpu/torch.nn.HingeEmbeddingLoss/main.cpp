#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create tensors and parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (should be same shape as input, with values -1 or 1)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            
            // HingeEmbeddingLoss expects target to be -1 or 1
            // We'll convert the target to have values -1 or 1
            target = torch::sign(target);
            
            // Replace zeros with ones (sign of zero is zero)
            target = torch::where(target == 0, torch::ones_like(target), target);
        } else {
            // If we don't have enough data, create a target with the same shape as input
            target = torch::sign(torch::randn_like(input));
            target = torch::where(target == 0, torch::ones_like(target), target);
        }
        
        // Parse margin parameter (default is 1.0)
        float margin = 1.0;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&margin, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Parse reduction parameter (0: none, 1: mean, 2: sum)
        int64_t reduction_mode = 1; // default to mean
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&reduction_mode, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            reduction_mode = std::abs(reduction_mode) % 3;
        }
        
        torch::nn::HingeEmbeddingLossOptions options;
        
        // Set margin
        options = options.margin(margin);
        
        // Set reduction
        switch (reduction_mode) {
            case 0:
                options = options.reduction(torch::kNone);
                break;
            case 1:
                options = options.reduction(torch::kMean);
                break;
            case 2:
                options = options.reduction(torch::kSum);
                break;
        }
        
        // Create the loss function
        torch::nn::HingeEmbeddingLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor loss = loss_fn(input, target);
        
        // Ensure loss is computed by accessing its value
        float loss_value = loss.item<float>();
        
        // Compute gradients
        if (loss.numel() > 0 && !loss.isnan().any().item<bool>() && 
            !loss.isinf().any().item<bool>()) {
            if (loss.requires_grad()) {
                loss.backward();
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
