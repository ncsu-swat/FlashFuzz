#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input and target tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the target tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get reduction mode from the input data
        torch::enumtype::kMean reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0:
                    reduction_mode = torch::kMean;
                    break;
                case 1:
                    reduction_mode = torch::kSum;
                    break;
                case 2:
                    reduction_mode = torch::kNone;
                    break;
            }
        }
        
        // Create MSELoss module with the selected reduction mode
        torch::nn::MSELossOptions options(reduction_mode);
        torch::nn::MSELoss mse_loss(options);
        
        // Apply MSELoss to the input and target tensors
        torch::Tensor loss = mse_loss(input, target);
        
        // Try to access the loss value to ensure computation is performed
        if (loss.defined()) {
            if (loss.numel() > 0) {
                float loss_value = loss.item<float>();
                (void)loss_value; // Prevent unused variable warning
            }
        }
        
        // Try backward pass if tensors require grad
        if (offset < Size && Data[offset] % 2 == 0) {
            // Create new tensors with requires_grad=true
            torch::Tensor input_grad = input.detach().clone().requires_grad_(true);
            torch::Tensor target_grad = target.detach().clone();
            
            // Recompute loss with gradients
            torch::Tensor loss_grad = mse_loss(input_grad, target_grad);
            
            // Perform backward pass if loss is scalar
            if (loss_grad.numel() == 1) {
                loss_grad.backward();
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