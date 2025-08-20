#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create query, key, value tensors
        torch::Tensor query = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor key = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor value = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create optional attn_mask tensor
        torch::Tensor attn_mask;
        bool use_attn_mask = false;
        if (offset < Size) {
            use_attn_mask = (Data[offset++] % 2 == 0);
            if (use_attn_mask && offset < Size) {
                attn_mask = fuzzer_utils::createTensor(Data, Size, offset);
            }
        }
        
        // Parse dropout_p
        double dropout_p = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&dropout_p, Data + offset, sizeof(double));
            offset += sizeof(double);
            dropout_p = std::abs(dropout_p);
            if (dropout_p > 1.0) {
                dropout_p = std::fmod(dropout_p, 1.0);
            }
        }
        
        // Parse is_causal
        bool is_causal = false;
        if (offset < Size) {
            is_causal = (Data[offset++] % 2 == 0);
        }
        
        // Parse scale
        double scale = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Call the SDPA kernel
        try {
            torch::Tensor result;
            if (use_attn_mask) {
                result = torch::scaled_dot_product_attention(
                    query, key, value, attn_mask, dropout_p, is_causal, scale);
            } else {
                result = torch::scaled_dot_product_attention(
                    query, key, value, {}, dropout_p, is_causal, scale);
            }
            
            // Perform some operation on the result to ensure it's used
            auto sum = result.sum();
            if (sum.numel() > 0) {
                volatile double val = sum.item<double>();
                (void)val;
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and shouldn't terminate fuzzing
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}