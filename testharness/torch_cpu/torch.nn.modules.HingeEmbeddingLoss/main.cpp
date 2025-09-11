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
            // If we don't have enough data for a second tensor, create a compatible target
            // Target should have same shape as input and contain values in {-1, 1}
            target = torch::ones_like(input);
            
            // Randomly negate some elements to get -1 values
            if (Size > 0) {
                auto mask = torch::randint(0, 2, input.sizes()) * 2 - 1;
                target = target * mask;
            }
        }
        
        // Extract margin parameter from the input data
        double margin = 1.0; // Default margin
        if (offset + sizeof(float) <= Size) {
            float margin_raw;
            std::memcpy(&margin_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure margin is a reasonable value
            margin = std::abs(margin_raw);
            if (std::isnan(margin) || std::isinf(margin)) {
                margin = 1.0;
            }
        }
        
        // Extract reduction parameter from the input data
        torch::nn::HingeEmbeddingLossOptions::reduction_t reduction = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0:
                    reduction = torch::kNone;
                    break;
                case 1:
                    reduction = torch::kMean;
                    break;
                case 2:
                    reduction = torch::kSum;
                    break;
            }
        }
        
        // Create HingeEmbeddingLoss module with the extracted parameters
        auto options = torch::nn::HingeEmbeddingLossOptions()
                           .margin(margin)
                           .reduction(reduction);
        auto loss_fn = torch::nn::HingeEmbeddingLoss(options);
        
        // Apply the loss function
        auto loss = loss_fn->forward(input, target);
        
        // Ensure the computation is executed
        loss.item<float>();
        
        // Try backward pass if the loss is a scalar
        if (loss.numel() == 1 && loss.requires_grad()) {
            loss.backward();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
