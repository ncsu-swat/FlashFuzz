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
        
        // Need at least 3 tensors for margin_ranking_loss: input1, input2, and target
        if (Size < 6) // Minimum bytes needed for basic tensor creation
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
        
        // Create target tensor (should be same shape as input1/input2)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a tensor with same shape as input1
            target = torch::ones_like(input1);
        }
        
        // Extract margin value from input data if available
        double margin = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&margin, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Extract reduction mode from input data if available
        int64_t reduction_raw = 1; // Default to mean
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&reduction_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Map reduction_raw to one of the valid reduction modes (0=none, 1=mean, 2=sum)
        int64_t reduction_mode = std::abs(reduction_raw) % 3;
        
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
        
        // Apply margin_ranking_loss
        torch::Tensor result = torch::margin_ranking_loss(
            input1, 
            input2, 
            target, 
            margin, 
            reduction
        );
        
        // Force computation to catch any errors
        result.item();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
