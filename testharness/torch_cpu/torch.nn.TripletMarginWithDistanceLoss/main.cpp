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
        torch::enumtype::Reduction reduction = torch::kMean;
        
        // If we have more data, use it to set options
        if (offset + 3 <= Size) {
            // Extract margin from data (between 0 and 10)
            margin = static_cast<double>(Data[offset]) / 25.0;
            offset++;
            
            // Extract swap from data (0 or 1)
            swap = Data[offset] % 2 == 1;
            offset++;
            
            // Extract reduction from data (0, 1, or 2)
            uint8_t red_val = Data[offset] % 3;
            if (red_val == 0) {
                reduction = torch::kNone;
            } else if (red_val == 1) {
                reduction = torch::kMean;
            } else {
                reduction = torch::kSum;
            }
            offset++;
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
            // Make tensors require grad
            anchor = anchor.detach().requires_grad_(true);
            positive = positive.detach().requires_grad_(true);
            negative = negative.detach().requires_grad_(true);
            
            // Recompute loss with grad-enabled tensors
            loss = loss_fn(anchor, positive, negative);
            
            // Backpropagate
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