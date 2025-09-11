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
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create SoftMarginLoss module with different configurations
        uint8_t config_byte = 0;
        if (offset < Size) {
            config_byte = Data[offset++];
        }
        
        // Parse reduction mode from config byte
        torch::nn::SoftMarginLossOptions::reduction_t reduction_mode;
        switch (config_byte % 3) {
            case 0:
                reduction_mode = torch::kNone;
                break;
            case 1:
                reduction_mode = torch::kMean;
                break;
            case 2:
                reduction_mode = torch::kSum;
                break;
            default:
                reduction_mode = torch::kMean;
        }
        
        // Create SoftMarginLoss module
        torch::nn::SoftMarginLossOptions options;
        options.reduction(reduction_mode);
        torch::nn::SoftMarginLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor output = loss_fn->forward(input, target);
        
        // Try backward pass if tensors have gradients enabled
        if (input.requires_grad()) {
            output.backward();
        }
        
        // Try with different reduction modes if we have enough data
        if (offset + 4 < Size) {
            // Create another SoftMarginLoss with different reduction
            torch::nn::SoftMarginLossOptions other_options;
            other_options.reduction(torch::kSum);
            torch::nn::SoftMarginLoss other_loss_fn(other_options);
            
            // Apply the other loss function
            torch::Tensor other_output = other_loss_fn->forward(input, target);
            
            // Try backward pass if tensors have gradients enabled
            if (input.requires_grad()) {
                other_output.backward();
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
