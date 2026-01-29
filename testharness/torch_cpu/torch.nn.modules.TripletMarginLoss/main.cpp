#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }
    
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors and parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create anchor tensor first
        torch::Tensor anchor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get the shape of anchor to create matching positive and negative tensors
        auto shape = anchor.sizes().vec();
        
        // Ensure anchor has at least 1 element
        if (anchor.numel() == 0) {
            return 0;
        }
        
        // Create positive and negative tensors with the same shape as anchor
        torch::Tensor positive = torch::randn(shape, torch::dtype(anchor.dtype()));
        torch::Tensor negative = torch::randn(shape, torch::dtype(anchor.dtype()));
        
        // Make tensors float for gradient computation
        anchor = anchor.to(torch::kFloat32).requires_grad_(true);
        positive = positive.to(torch::kFloat32).requires_grad_(true);
        negative = negative.to(torch::kFloat32).requires_grad_(true);
        
        // Get parameters for TripletMarginLoss from remaining data
        double margin = 1.0;
        double p = 2.0;
        double eps = 1e-6;
        bool swap = false;
        int reduction_mode = 1; // 0=none, 1=mean, 2=sum
        
        if (offset + sizeof(uint16_t) <= Size) {
            uint16_t margin_raw;
            std::memcpy(&margin_raw, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            margin = static_cast<double>(margin_raw) / 1000.0;
        }
        
        if (offset < Size) {
            // Extract p value (norm) - keep it reasonable (1.0 to 3.0)
            p = 1.0 + static_cast<double>(Data[offset++] % 20) / 10.0;
        }
        
        if (offset < Size) {
            // Extract eps
            eps = 1e-9 + static_cast<double>(Data[offset++]) / 1e10;
        }
        
        if (offset < Size) {
            swap = Data[offset++] % 2 == 1;
        }
        
        if (offset < Size) {
            reduction_mode = Data[offset++] % 3;
        }
        
        // Create TripletMarginLoss module with options
        torch::nn::TripletMarginLossOptions options;
        options.margin(margin)
               .p(p)
               .eps(eps)
               .swap(swap);
        
        switch (reduction_mode) {
            case 0: options.reduction(torch::kNone); break;
            case 1: options.reduction(torch::kMean); break;
            case 2: options.reduction(torch::kSum); break;
            default: options.reduction(torch::kMean);
        }
        
        auto triplet_loss = torch::nn::TripletMarginLoss(options);
        
        // Apply the loss function
        torch::Tensor loss = triplet_loss->forward(anchor, positive, negative);
        
        // Ensure computation is completed - handle both scalar and non-scalar cases
        if (reduction_mode == 0) {
            // reduction="none" returns tensor of same shape
            loss.sum().item<float>();
        } else {
            // reduction="mean" or "sum" returns scalar
            loss.item<float>();
        }
        
        // Also test backward pass for better coverage
        try {
            if (reduction_mode != 0) {
                loss.backward();
            } else {
                loss.sum().backward();
            }
        } catch (...) {
            // Backward pass may fail in some edge cases, that's ok
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}