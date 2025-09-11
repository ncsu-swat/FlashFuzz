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
        if (Size < 10) {
            return 0;
        }
        
        // Create anchor, positive, and negative tensors
        torch::Tensor anchor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for more tensors
        if (offset >= Size - 5) {
            return 0;
        }
        
        torch::Tensor positive = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size - 5) {
            return 0;
        }
        
        torch::Tensor negative = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse options for TripletMarginWithDistanceLoss
        double margin = 1.0;
        bool swap = false;
        torch::nn::TripletMarginWithDistanceLossOptions::reduction_t reduction = torch::kMean;
        
        // If we have more data, use it to set options
        if (offset + 3 <= Size) {
            // Extract margin from data (convert to a reasonable range)
            uint8_t margin_byte = Data[offset++];
            margin = static_cast<double>(margin_byte) / 10.0; // Range: 0.0 to 25.5
            
            // Extract swap flag
            swap = Data[offset++] % 2 == 1;
            
            // Extract reduction mode
            uint8_t reduction_byte = Data[offset++] % 3;
            switch (reduction_byte) {
                case 0: reduction = torch::kNone; break;
                case 1: reduction = torch::kMean; break;
                case 2: reduction = torch::kSum; break;
                default: reduction = torch::kMean;
            }
        }
        
        // Create the TripletMarginWithDistanceLoss module
        auto options = torch::nn::TripletMarginWithDistanceLossOptions()
            .margin(margin)
            .swap(swap)
            .reduction(reduction);
        
        torch::nn::TripletMarginWithDistanceLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor loss = loss_fn(anchor, positive, negative);
        
        // Ensure computation is performed
        loss.item<float>();
        
        // Test backward pass if tensors require grad
        if (offset < Size && Data[offset] % 2 == 0) {
            // Create new tensors with requires_grad=true
            auto anchor_grad = anchor.detach().clone().requires_grad_(true);
            auto positive_grad = positive.detach().clone().requires_grad_(true);
            auto negative_grad = negative.detach().clone().requires_grad_(true);
            
            // Compute loss and backward
            auto loss_grad = loss_fn(anchor_grad, positive_grad, negative_grad);
            loss_grad.backward();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
