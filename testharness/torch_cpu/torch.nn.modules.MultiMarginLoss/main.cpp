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
            // If we don't have enough data for a second tensor, create a simple one
            target = torch::zeros_like(input).to(torch::kLong);
        }
        
        // Extract parameters for MultiMarginLoss
        float p = 1.0;
        float margin = 1.0;
        torch::Tensor weight;
        torch::nn::MultiMarginLossOptions::reduction_t reduction = torch::kMean;
        
        // If we have more data, use it to set parameters
        if (offset + 4 <= Size) {
            // Extract p value (1 or 2)
            p = (Data[offset++] % 2) + 1;
            
            // Extract margin value
            uint32_t margin_bits;
            std::memcpy(&margin_bits, Data + offset, sizeof(margin_bits));
            offset += sizeof(margin_bits);
            margin = static_cast<float>(margin_bits) / 1000.0f;
            
            // Create weight tensor if we have more data
            if (offset < Size) {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
            }
            
            // Extract reduction mode
            if (offset < Size) {
                int reduction_mode = Data[offset++] % 3;
                if (reduction_mode == 0) {
                    reduction = torch::kNone;
                } else if (reduction_mode == 1) {
                    reduction = torch::kMean;
                } else {
                    reduction = torch::kSum;
                }
            }
        }
        
        // Create MultiMarginLoss module
        torch::nn::MultiMarginLossOptions options;
        options.p(p).margin(margin).reduction(reduction);
        
        if (weight.defined()) {
            options.weight(weight);
        }
        
        auto loss_fn = torch::nn::MultiMarginLoss(options);
        
        // Apply the loss function
        torch::Tensor output = loss_fn->forward(input, target);
        
        // Optionally compute gradients
        if (offset < Size && Data[offset] % 2 == 0) {
            if (output.numel() > 0 && output.requires_grad()) {
                output.backward();
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
