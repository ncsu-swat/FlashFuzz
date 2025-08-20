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
        
        // Create weight tensor (embedding table)
        torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure weight has at least 2 dimensions for embedding_bag
        if (weight.dim() < 2) {
            if (weight.dim() == 0) {
                weight = weight.unsqueeze(0).unsqueeze(0);
            } else {
                weight = weight.unsqueeze(0);
            }
        }
        
        // Create indices tensor
        torch::Tensor indices;
        if (offset < Size) {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure indices are integers
            indices = indices.to(torch::kInt64);
            
            // Clamp indices to valid range for the weight tensor
            if (weight.size(0) > 0) {
                indices = torch::clamp(indices, 0, weight.size(0) - 1);
            }
        } else {
            // Create default indices if we've run out of data
            indices = torch::tensor({0, 1, 2}, torch::kInt64);
        }
        
        // Create offsets tensor
        torch::Tensor offsets;
        if (offset < Size) {
            offsets = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure offsets are integers
            offsets = offsets.to(torch::kInt64);
            
            // Ensure offsets are sorted and within valid range
            if (offsets.numel() > 0) {
                offsets = std::get<0>(torch::sort(offsets));
                if (indices.numel() > 0) {
                    offsets = torch::clamp(offsets, 0, indices.numel());
                }
            }
        } else {
            // Create default offsets if we've run out of data
            offsets = torch::tensor({0}, torch::kInt64);
        }
        
        // Get mode parameter (0=sum, 1=mean, 2=max)
        int64_t mode = 0;
        if (offset < Size) {
            mode = static_cast<int64_t>(Data[offset++]) % 3;
        }
        
        // Get sparse parameter
        bool sparse = false;
        if (offset < Size) {
            sparse = static_cast<bool>(Data[offset++] & 1);
        }
        
        // Get scale_grad_by_freq parameter
        bool scale_grad_by_freq = false;
        if (offset < Size) {
            scale_grad_by_freq = static_cast<bool>(Data[offset++] & 1);
        }
        
        // Get padding_idx parameter
        std::optional<int64_t> padding_idx = std::nullopt;
        if (offset < Size) {
            uint8_t padding_selector = Data[offset++];
            if (padding_selector & 1) {
                // Use a valid padding_idx
                if (weight.size(0) > 0) {
                    padding_idx = padding_selector % weight.size(0);
                }
            }
        }
        
        // Apply embedding_bag operation
        auto result = torch::embedding_bag(
            weight,
            indices,
            offsets,
            scale_grad_by_freq,
            mode,
            sparse,
            std::nullopt,  // per_sample_weights
            false,         // include_last_offset
            padding_idx
        );
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}