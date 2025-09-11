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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor with same shape as input
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a target with the same shape as input
            target = torch::ones_like(input);
        }
        
        // Create reduction mode from remaining data
        torch::enumtype::kMean reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_selector = Data[offset++] % 3;
            if (reduction_selector == 0) {
                reduction_mode = torch::kNone;
            } else if (reduction_selector == 1) {
                reduction_mode = torch::kMean;
            } else {
                reduction_mode = torch::kSum;
            }
        }
        
        // Create SoftMarginLoss module with the selected reduction mode
        torch::nn::SoftMarginLossOptions options(reduction_mode);
        torch::nn::SoftMarginLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor output = loss_fn(input, target);
        
        // Try backward pass if possible
        if (output.numel() > 0 && output.requires_grad()) {
            output.backward();
        }
        
        // Try with default options
        torch::nn::SoftMarginLoss default_loss_fn;
        torch::Tensor default_output = default_loss_fn(input, target);
        
        // Try direct functional version
        torch::Tensor functional_output = torch::nn::functional::soft_margin_loss(
            input, target, torch::nn::functional::SoftMarginLossFuncOptions().reduction(reduction_mode));
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
