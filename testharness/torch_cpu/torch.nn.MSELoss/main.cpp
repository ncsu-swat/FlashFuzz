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
        
        // Create input and target tensors for MSELoss
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the target tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get reduction mode from the remaining data
        torch::nn::MSELossOptions options;
        if (offset < Size) {
            uint8_t reduction_selector = Data[offset++] % 3;
            switch (reduction_selector) {
                case 0:
                    options.reduction(torch::kNone);
                    break;
                case 1:
                    options.reduction(torch::kMean);
                    break;
                case 2:
                    options.reduction(torch::kSum);
                    break;
            }
        } else {
            options.reduction(torch::kMean);
        }
        
        // Create MSELoss module with the selected reduction mode
        torch::nn::MSELoss mse_loss(options);
        
        // Apply MSELoss to the input and target tensors
        torch::Tensor loss = mse_loss(input, target);
        
        // Ensure computation is completed
        loss.item<float>();
        
        // Try backward pass if tensors require grad
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Create new tensors with requires_grad=true
            torch::Tensor input_with_grad = input.detach().clone().set_requires_grad(true);
            torch::Tensor target_with_grad = target.detach().clone();
            
            // Recompute loss with gradients
            torch::Tensor loss_with_grad = mse_loss(input_with_grad, target_with_grad);
            
            // Backward pass
            loss_with_grad.backward();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
