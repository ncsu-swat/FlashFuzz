#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create query, key, value tensors
        torch::Tensor query = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have enough data left for the next tensors
        if (offset >= Size - 5) {
            return 0;
        }
        
        torch::Tensor key = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size - 5) {
            return 0;
        }
        
        torch::Tensor value = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse embed_dim and num_heads from remaining data
        if (offset + 2 <= Size) {
            int64_t embed_dim = static_cast<int64_t>(Data[offset++]) + 1; // Ensure non-zero
            int64_t num_heads = static_cast<int64_t>(Data[offset++]) + 1; // Ensure non-zero
            
            // Create MultiheadAttention module
            torch::nn::MultiheadAttention mha(
                torch::nn::MultiheadAttentionOptions(embed_dim, num_heads)
            );
            
            // Parse additional options if we have more data
            if (offset + 3 <= Size) {
                bool add_bias_kv = Data[offset++] % 2 == 0;
                bool add_zero_attn = Data[offset++] % 2 == 0;
                double dropout_p = static_cast<double>(Data[offset++]) / 255.0;
                
                mha = torch::nn::MultiheadAttention(
                    torch::nn::MultiheadAttentionOptions(embed_dim, num_heads)
                        .bias(true)
                        .add_bias_kv(add_bias_kv)
                        .add_zero_attn(add_zero_attn)
                        .dropout(dropout_p)
                );
            }
            
            // Parse key_padding_mask and attn_mask if we have more data
            torch::Tensor key_padding_mask;
            torch::Tensor attn_mask;
            
            if (offset < Size - 5) {
                key_padding_mask = fuzzer_utils::createTensor(Data, Size, offset);
            }
            
            if (offset < Size - 5) {
                attn_mask = fuzzer_utils::createTensor(Data, Size, offset);
            }
            
            // Apply MultiheadAttention
            try {
                auto result = mha->forward(query, key, value, key_padding_mask, true, attn_mask);
                
                // Access the result to ensure computation is performed
                auto attn_output = std::get<0>(result);
                auto attn_weights = std::get<1>(result);
                
                // Perform some operation on the output to ensure it's used
                auto sum = attn_output.sum();
                if (attn_weights.defined()) {
                    sum += attn_weights.sum();
                }
            } catch (const c10::Error& e) {
                // PyTorch-specific exceptions are expected and not a fuzzer error
                return 0;
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