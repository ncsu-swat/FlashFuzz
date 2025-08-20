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
        if (Size < 3) {
            return 0;
        }
        
        // Create anchor, positive, and negative tensors
        torch::Tensor anchor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for positive tensor
        if (offset >= Size) {
            return 0;
        }
        torch::Tensor positive = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for negative tensor
        if (offset >= Size) {
            return 0;
        }
        torch::Tensor negative = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get parameters for TripletMarginLoss from remaining data
        double margin = 1.0;
        double p = 2.0;
        bool swap = false;
        std::string reduction = "mean";
        
        if (offset + 2 < Size) {
            // Extract margin from data (convert to a reasonable range)
            uint16_t margin_raw;
            std::memcpy(&margin_raw, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            margin = static_cast<double>(margin_raw) / 1000.0; // Scale to a reasonable range
        }
        
        if (offset < Size) {
            // Extract p value (norm) from data
            p = static_cast<double>(Data[offset++]) / 10.0 + 0.1; // Ensure p is positive
        }
        
        if (offset < Size) {
            // Extract swap boolean
            swap = Data[offset++] % 2 == 1;
        }
        
        if (offset < Size) {
            // Extract reduction mode
            uint8_t reduction_selector = Data[offset++] % 3;
            switch (reduction_selector) {
                case 0: reduction = "none"; break;
                case 1: reduction = "mean"; break;
                case 2: reduction = "sum"; break;
                default: reduction = "mean";
            }
        }
        
        // Create TripletMarginLoss module
        torch::nn::TripletMarginLossOptions options;
        options.margin(margin)
               .p(p)
               .swap(swap)
               .reduction(torch::kMean);
        
        if (reduction == "none") {
            options.reduction(torch::kNone);
        } else if (reduction == "sum") {
            options.reduction(torch::kSum);
        }
        
        auto triplet_loss = torch::nn::TripletMarginLoss(options);
        
        // Apply the loss function
        torch::Tensor loss = triplet_loss->forward(anchor, positive, negative);
        
        // Ensure computation is completed
        loss.item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}