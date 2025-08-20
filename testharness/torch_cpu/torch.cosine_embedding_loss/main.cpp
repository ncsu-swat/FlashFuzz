#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors for cosine_embedding_loss
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (should be 1D tensor with values -1 or 1)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure target has values -1 or 1 (required by cosine_embedding_loss)
            // We don't do this as a sanity check, but rather to make valid inputs for the API
            target = torch::sign(target);
            
            // Handle 0 values in target (sign(0) = 0, but we need -1 or 1)
            target = torch::where(target == 0, torch::ones_like(target), target);
        } else {
            // If we don't have enough data, create a simple target tensor
            target = torch::ones({1});
        }
        
        // Get margin parameter from the remaining data
        float margin = 0.0;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&margin, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Get reduction mode from the remaining data
        int64_t reduction_mode = 1; // Default to mean
        if (offset < Size) {
            reduction_mode = static_cast<int64_t>(Data[offset++]) % 3;
        }
        
        // Map reduction_mode to at::Reduction
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
        
        // Apply cosine_embedding_loss
        torch::Tensor loss = torch::cosine_embedding_loss(
            input1, 
            input2, 
            target, 
            margin, 
            reduction
        );
        
        // Ensure the loss is computed by accessing its value
        float loss_value = 0.0;
        if (loss.numel() > 0) {
            if (loss.dim() == 0) {
                loss_value = loss.item<float>();
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