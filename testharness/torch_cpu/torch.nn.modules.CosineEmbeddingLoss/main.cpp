#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create tensors and parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have more data for the second tensor
        if (offset >= Size - 2) {
            return 0;
        }
        
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (should be 1D tensor with values -1 or 1)
        torch::Tensor target;
        if (offset < Size - 2) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure target has values -1 or 1 as required by CosineEmbeddingLoss
            target = torch::sign(target);
            
            // If sign returns 0 for any elements, replace with 1
            target = torch::where(target == 0, torch::ones_like(target), target);
        } else {
            // Create a default target if not enough data
            target = torch::ones({std::max<int64_t>(1, input1.size(0))});
        }
        
        // Get margin parameter from the input data
        float margin = 0.0;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&margin, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Get reduction mode from the input data
        int64_t reduction_raw = 1; // Default to mean
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&reduction_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Map to valid reduction modes (0=none, 1=mean, 2=sum)
        reduction_raw = std::abs(reduction_raw) % 3;
        torch::nn::CosineEmbeddingLossOptions::reduction_t reduction;
        
        if (reduction_raw == 0) {
            reduction = torch::kNone;
        } else if (reduction_raw == 1) {
            reduction = torch::kMean;
        } else {
            reduction = torch::kSum;
        }
        
        // Create the CosineEmbeddingLoss module
        torch::nn::CosineEmbeddingLossOptions options;
        options.margin(margin);
        options.reduction(reduction);
        torch::nn::CosineEmbeddingLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor output = loss_fn->forward(input1, input2, target);
        
        // Ensure the computation is executed
        output.item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}