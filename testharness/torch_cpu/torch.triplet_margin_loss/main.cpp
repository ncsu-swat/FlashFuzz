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
        
        // Need at least 3 tensors for triplet_margin_loss: anchor, positive, negative
        if (Size < 6) // Minimum bytes needed for basic tensor creation
            return 0;
        
        // Create anchor tensor
        torch::Tensor anchor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create positive tensor
        if (offset >= Size)
            return 0;
        torch::Tensor positive = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create negative tensor
        if (offset >= Size)
            return 0;
        torch::Tensor negative = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse margin parameter (default is 1.0)
        double margin = 1.0;
        if (offset + sizeof(float) <= Size) {
            float margin_raw;
            std::memcpy(&margin_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            margin = static_cast<double>(margin_raw);
        }
        
        // Parse p parameter (default is 2.0)
        double p = 2.0;
        if (offset + sizeof(float) <= Size) {
            float p_raw;
            std::memcpy(&p_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            p = static_cast<double>(p_raw);
        }
        
        // Parse eps parameter (default is 1e-6)
        double eps = 1e-6;
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            eps = static_cast<double>(eps_raw);
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
        if (result.defined()) {
            volatile float sum = result.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
