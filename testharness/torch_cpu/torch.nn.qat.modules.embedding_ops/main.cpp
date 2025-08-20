#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor indices = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get parameters for embedding
        int64_t padding_idx = -1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_idx, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Try basic embedding operations
        try {
            auto result1 = torch::nn::functional::embedding(indices, weight, padding_idx);
        } catch (const std::exception&) {
            // Catch and continue
        }
        
        try {
            // Create additional tensors for embedding_bag
            torch::Tensor offsets = fuzzer_utils::createTensor(Data, Size, offset);
            
            auto result2 = torch::nn::functional::embedding_bag(
                indices, 
                weight,
                offsets
            );
        } catch (const std::exception&) {
            // Catch and continue
        }
        
        // Try with different modes
        uint8_t mode_byte = 0;
        if (offset < Size) {
            mode_byte = Data[offset++];
        }
        int64_t mode = mode_byte % 3; // 0=sum, 1=mean, 2=max
        
        try {
            auto result3 = torch::nn::functional::embedding_bag(
                indices, 
                weight,
                torch::Tensor(), // offsets
                torch::Tensor(), // max_norm
                2.0,             // norm_type
                false,           // scale_grad_by_freq
                mode,            // mode
                false,           // sparse
                torch::Tensor(), // per_sample_weights
                false            // include_last_offset
            );
        } catch (const std::exception&) {
            // Catch and continue
        }
        
        // Try with different sparse values
        bool sparse = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        
        try {
            auto result4 = torch::nn::functional::embedding_bag(
                indices, 
                weight,
                torch::Tensor(), // offsets
                torch::Tensor(), // max_norm
                2.0,             // norm_type
                false,           // scale_grad_by_freq
                0,               // mode (sum)
                sparse,          // sparse
                torch::Tensor(), // per_sample_weights
                false            // include_last_offset
            );
        } catch (const std::exception&) {
            // Catch and continue
        }
        
        // Try with include_last_offset
        bool include_last_offset = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        
        try {
            auto result5 = torch::nn::functional::embedding_bag(
                indices, 
                weight,
                torch::Tensor(), // offsets
                torch::Tensor(), // max_norm
                2.0,             // norm_type
                false,           // scale_grad_by_freq
                0,               // mode (sum)
                false,           // sparse
                torch::Tensor(), // per_sample_weights
                include_last_offset
            );
        } catch (const std::exception&) {
            // Catch and continue
        }
        
        // Try with per_sample_weights if there's enough data
        try {
            if (offset < Size) {
                torch::Tensor per_sample_weights = fuzzer_utils::createTensor(Data, Size, offset);
                auto result6 = torch::nn::functional::embedding_bag(
                    indices, 
                    weight,
                    torch::Tensor(), // offsets
                    torch::Tensor(), // max_norm
                    2.0,             // norm_type
                    false,           // scale_grad_by_freq
                    0,               // mode (sum)
                    false,           // sparse
                    per_sample_weights,
                    false            // include_last_offset
                );
            }
        } catch (const std::exception&) {
            // Catch and continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}