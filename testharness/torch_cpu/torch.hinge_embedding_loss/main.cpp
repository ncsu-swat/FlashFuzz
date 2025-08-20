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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor with same shape as input
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract margin parameter from the input data
        double margin = 1.0;
        if (offset + sizeof(float) <= Size) {
            float margin_val;
            std::memcpy(&margin_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            margin = static_cast<double>(margin_val);
        }
        
        // Extract reduction parameter from the input data
        int64_t reduction_raw = 1; // Default to mean
        if (offset + sizeof(int8_t) <= Size) {
            int8_t reduction_val;
            std::memcpy(&reduction_val, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            reduction_raw = static_cast<int64_t>(reduction_val) % 3;
        }
        
        // Map reduction value to at::Reduction enum
        int64_t reduction;
        switch (reduction_raw) {
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
        
        // Apply hinge_embedding_loss
        torch::Tensor result = torch::hinge_embedding_loss(input, target, margin, reduction);
        
        // Ensure the result is valid by accessing its data
        result.item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}