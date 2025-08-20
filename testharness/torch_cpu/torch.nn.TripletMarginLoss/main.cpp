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
        if (Size < 10) {
            return 0;
        }
        
        // Create anchor, positive, and negative tensors
        torch::Tensor anchor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for positive and negative tensors
        if (offset >= Size - 5) {
            return 0;
        }
        
        torch::Tensor positive = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size - 5) {
            return 0;
        }
        
        torch::Tensor negative = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse margin parameter (if we have data left)
        double margin = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&margin, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse p parameter (if we have data left)
        double p = 2.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure p is valid (p >= 0)
            p = std::abs(p);
        }
        
        // Parse swap parameter (if we have data left)
        bool swap = false;
        if (offset < Size) {
            swap = static_cast<bool>(Data[offset] & 0x01);
            offset++;
        }
        
        // Parse reduction parameter (if we have data left)
        torch::nn::TripletMarginLossOptions::reduction_t reduction = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_val = Data[offset] % 3;
            offset++;
            
            switch (reduction_val) {
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
        
        // Create TripletMarginLoss module with the parsed parameters
        auto options = torch::nn::TripletMarginLossOptions()
            .margin(margin)
            .p(p)
            .swap(swap)
            .reduction(reduction);
        
        auto triplet_loss = torch::nn::TripletMarginLoss(options);
        
        // Apply the loss function
        auto loss = triplet_loss->forward(anchor, positive, negative);
        
        // Ensure the loss is computed by accessing its value
        float loss_value = loss.item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}