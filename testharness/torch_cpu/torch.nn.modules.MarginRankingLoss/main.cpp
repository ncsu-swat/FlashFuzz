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
        
        // Create input tensors for MarginRankingLoss
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the target tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get margin value from the remaining data
        float margin = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&margin, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Get reduction mode from the remaining data
        uint8_t reduction_byte = 0;
        if (offset < Size) {
            reduction_byte = Data[offset++];
        }
        
        // Map reduction_byte to one of the three reduction modes
        torch::nn::MarginRankingLossOptions::reduction_t reduction_mode;
        switch (reduction_byte % 3) {
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
                reduction_mode = torch::kMean; // Default
        }
        
        // Create MarginRankingLoss module with the parsed options
        auto options = torch::nn::MarginRankingLossOptions()
                           .margin(margin)
                           .reduction(reduction_mode);
        
        torch::nn::MarginRankingLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor loss = loss_fn(input1, input2, target);
        
        // Perform a backward pass if the tensor requires grad
        if (loss.requires_grad()) {
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