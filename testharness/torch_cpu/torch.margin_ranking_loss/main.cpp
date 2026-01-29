#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }
    
    try
    {
        size_t offset = 0;
        
        // Need at least some bytes for tensor creation
        if (Size < 6)
            return 0;
        
        // Create input1 tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create input2 tensor with same shape as input1 if possible
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a tensor with same shape as input1
            input2 = torch::ones_like(input1);
        }
        
        // Ensure input1 and input2 have compatible shapes by broadcasting
        try {
            // Try to broadcast to common shape
            auto broadcasted = torch::broadcast_tensors({input1, input2});
            input1 = broadcasted[0];
            input2 = broadcasted[1];
        } catch (...) {
            // If broadcasting fails, make input2 same shape as input1
            input2 = torch::randn_like(input1);
        }
        
        // Create target tensor with values -1 or 1 (required by margin_ranking_loss)
        torch::Tensor target;
        if (offset < Size) {
            // Use fuzzer data to determine target values
            target = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert to -1 or 1 based on sign
            target = torch::where(target >= 0, 
                                  torch::ones_like(target), 
                                  -torch::ones_like(target));
        } else {
            target = torch::ones_like(input1);
        }
        
        // Ensure target has same shape as inputs
        try {
            auto broadcasted = torch::broadcast_tensors({input1, target});
            target = broadcasted[1];
        } catch (...) {
            target = torch::ones_like(input1);
        }
        
        // Extract margin value from input data if available
        double margin = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&margin, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize margin to avoid NaN/Inf issues
            if (std::isnan(margin) || std::isinf(margin)) {
                margin = 0.0;
            }
            // Clamp margin to reasonable range
            margin = std::max(-100.0, std::min(100.0, margin));
        }
        
        // Extract reduction mode from input data if available
        uint8_t reduction_raw = 1; // Default to mean
        if (offset < Size) {
            reduction_raw = Data[offset];
            offset++;
        }
        
        // Map reduction_raw to one of the valid reduction modes (0=none, 1=mean, 2=sum)
        int64_t reduction_mode = reduction_raw % 3;
        
        // Convert reduction_mode to at::Reduction
        int64_t reduction;
        switch (reduction_mode) {
            case 0:
                reduction = at::Reduction::None;
                break;
            case 1:
                reduction = at::Reduction::Mean;
                break;
            case 2:
                reduction = at::Reduction::Sum;
                break;
            default:
                reduction = at::Reduction::Mean;
        }
        
        // Ensure inputs are float type for margin_ranking_loss
        if (!input1.is_floating_point()) {
            input1 = input1.to(torch::kFloat32);
        }
        if (!input2.is_floating_point()) {
            input2 = input2.to(torch::kFloat32);
        }
        if (!target.is_floating_point()) {
            target = target.to(torch::kFloat32);
        }
        
        // Apply margin_ranking_loss
        torch::Tensor result = torch::margin_ranking_loss(
            input1, 
            input2, 
            target, 
            margin, 
            reduction
        );
        
        // Force computation - use sum() which works regardless of tensor shape
        volatile float check = result.sum().item<float>();
        (void)check;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}