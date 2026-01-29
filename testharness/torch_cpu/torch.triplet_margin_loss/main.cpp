#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For isnan, isinf

// --- Fuzzer Entry Point ---
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
        
        // Need at least some bytes for tensor creation and parameters
        if (Size < 6)
            return 0;
        
        // Create anchor tensor
        torch::Tensor anchor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create positive and negative tensors with the same shape as anchor
        // This ensures shape compatibility required by triplet_margin_loss
        if (offset >= Size)
            return 0;
        torch::Tensor positive = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size)
            return 0;
        torch::Tensor negative = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to make tensors compatible by reshaping to common size
        try {
            auto target_sizes = anchor.sizes().vec();
            if (positive.numel() > 0 && negative.numel() > 0) {
                positive = positive.flatten().narrow(0, 0, std::min(positive.numel(), anchor.numel()));
                negative = negative.flatten().narrow(0, 0, std::min(negative.numel(), anchor.numel()));
                anchor = anchor.flatten().narrow(0, 0, std::min({anchor.numel(), positive.numel(), negative.numel()}));
                
                int64_t common_size = anchor.numel();
                if (common_size == 0)
                    return 0;
                    
                positive = positive.narrow(0, 0, common_size);
                negative = negative.narrow(0, 0, common_size);
            }
        } catch (...) {
            // Shape manipulation failed, skip this input
            return 0;
        }
        
        // Ensure tensors are float type for loss computation
        anchor = anchor.to(torch::kFloat32);
        positive = positive.to(torch::kFloat32);
        negative = negative.to(torch::kFloat32);
        
        // Parse margin parameter (default is 1.0)
        double margin = 1.0;
        if (offset + sizeof(float) <= Size) {
            float margin_raw;
            std::memcpy(&margin_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (!std::isnan(margin_raw) && !std::isinf(margin_raw)) {
                margin = static_cast<double>(margin_raw);
            }
        }
        
        // Parse p parameter (default is 2.0, must be positive)
        double p = 2.0;
        if (offset + sizeof(float) <= Size) {
            float p_raw;
            std::memcpy(&p_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (!std::isnan(p_raw) && !std::isinf(p_raw) && p_raw > 0.0f) {
                p = static_cast<double>(p_raw);
            }
        }
        
        // Parse eps parameter (default is 1e-6, must be non-negative)
        double eps = 1e-6;
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (!std::isnan(eps_raw) && !std::isinf(eps_raw) && eps_raw >= 0.0f) {
                eps = static_cast<double>(eps_raw);
            }
        }
        
        // Parse swap parameter (default is false)
        bool swap = false;
        if (offset < Size) {
            swap = Data[offset++] & 0x1;
        }
        
        // Parse reduction parameter (default is Mean)
        int64_t reduction = at::Reduction::Mean;
        if (offset < Size) {
            uint8_t reduction_raw = Data[offset++] % 3;
            switch (reduction_raw) {
                case 0: reduction = at::Reduction::None; break;
                case 1: reduction = at::Reduction::Mean; break;
                case 2: reduction = at::Reduction::Sum; break;
            }
        }
        
        // Apply triplet_margin_loss
        torch::Tensor result = torch::triplet_margin_loss(
            anchor, positive, negative, margin, p, eps, swap, reduction);
        
        // Ensure the result is used to prevent optimization
        if (result.defined() && result.numel() > 0) {
            volatile float sum = result.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}