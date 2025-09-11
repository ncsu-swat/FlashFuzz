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
        
        // Need at least a few bytes to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors for CosineEmbeddingLoss
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (should be 1D with values of 1 or -1)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure target has values of 1 or -1
            target = torch::sign(target);
            
            // Handle case where sign returns 0 (convert to 1)
            target = torch::where(target == 0, torch::ones_like(target), target);
        } else {
            // If we don't have enough data, create a simple target
            target = torch::ones({std::max<int64_t>(1, input1.size(0))});
        }
        
        // Get reduction mode from the data
        torch::nn::CosineEmbeddingLossOptions::reduction_t reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_selector = Data[offset++] % 3;
            switch (reduction_selector) {
                case 0:
                    reduction_mode = torch::kNone;
                    break;
                case 1:
                    reduction_mode = torch::kSum;
                    break;
                default:
                    reduction_mode = torch::kMean;
                    break;
            }
        }
        
        // Get margin value from the data
        double margin = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&margin, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Create CosineEmbeddingLoss module
        auto options = torch::nn::CosineEmbeddingLossOptions()
                          .margin(margin)
                          .reduction(reduction_mode);
        
        torch::nn::CosineEmbeddingLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor loss = loss_fn(input1, input2, target);
        
        // Ensure loss is computed and force evaluation
        float loss_value = loss.item<float>();
        (void)loss_value; // Suppress unused variable warning
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
