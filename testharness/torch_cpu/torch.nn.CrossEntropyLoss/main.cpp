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
        
        // Create input tensor (logits)
        torch::Tensor logits = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (labels)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, create a compatible target
            if (logits.dim() >= 2) {
                // For typical case: [batch_size, num_classes, ...]
                int64_t batch_size = logits.size(0);
                target = torch::randint(0, logits.size(1), {batch_size});
            } else if (logits.dim() == 1) {
                // For 1D case, create a single target
                target = torch::randint(0, logits.size(0), {1});
            } else {
                // For 0D case, create a scalar target
                target = torch::tensor(0);
            }
        }
        
        // Parse weight parameter (optional)
        torch::Tensor weight;
        if (offset + 1 < Size) {
            uint8_t use_weight = Data[offset++];
            if (use_weight % 2 == 1 && offset < Size) {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
            }
        }
        
        // Parse reduction method
        torch::kMean reduction = torch::kMean;
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
        
        // Parse ignore_index
        int64_t ignore_index = -100; // Default value
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&ignore_index, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse label_smoothing
        double label_smoothing = 0.0; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&label_smoothing, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure label_smoothing is in valid range [0.0, 1.0]
            label_smoothing = std::abs(label_smoothing);
            if (label_smoothing > 1.0) {
                label_smoothing = std::fmod(label_smoothing, 1.0);
            }
        }
        
        // Create CrossEntropyLoss with the parsed parameters
        torch::nn::CrossEntropyLoss criterion(
            torch::nn::CrossEntropyLossOptions()
                .weight(weight)
                .ignore_index(ignore_index)
                .reduction(reduction)
                .label_smoothing(label_smoothing)
        );
        
        // Apply the loss function
        torch::Tensor loss = criterion(logits, target);
        
        // Optionally compute gradients
        if (offset < Size && Data[offset++] % 2 == 1) {
            if (loss.numel() == 1) {
                loss.backward();
            } else if (loss.numel() > 1) {
                // For reduction=none, we need a gradient of matching shape
                torch::Tensor grad = torch::ones_like(loss);
                loss.backward(grad);
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
