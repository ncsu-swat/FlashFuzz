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
        
        // Parse parameters for MultiheadAttention
        int64_t embed_dim = (Data[0] % 16 + 1) * 8;  // Ensure it's a multiple of 8 and non-zero
        int64_t num_heads = Data[1] % 8 + 1;         // Between 1 and 8 heads
        double dropout = static_cast<double>(Data[2]) / 255.0;  // Between 0 and 1
        bool bias = Data[3] % 2 == 0;                // Random boolean
        bool add_bias_kv = Data[4] % 2 == 0;         // Random boolean
        bool add_zero_attn = Data[5] % 2 == 0;       // Random boolean
        
        // Create the MultiheadAttention module
        torch::nn::MultiheadAttention mha(
            torch::nn::MultiheadAttentionOptions(embed_dim, num_heads)
                .dropout(dropout)
                .bias(bias)
                .add_bias_kv(add_bias_kv)
                .add_zero_attn(add_zero_attn)
        );
        
        // Create input tensors
        offset = 7;  // Start after the parameters
        
        // Create query tensor
        torch::Tensor query;
        if (offset < Size) {
            query = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default tensor if we don't have enough data
            int64_t seq_len = 10;
            int64_t batch_size = 2;
            query = torch::rand({seq_len, batch_size, embed_dim});
        }
        
        // Create key tensor
        torch::Tensor key;
        if (offset < Size) {
            key = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default tensor if we don't have enough data
            int64_t seq_len = 10;
            int64_t batch_size = 2;
            key = torch::rand({seq_len, batch_size, embed_dim});
        }
        
        // Create value tensor
        torch::Tensor value;
        if (offset < Size) {
            value = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default tensor if we don't have enough data
            int64_t seq_len = 10;
            int64_t batch_size = 2;
            value = torch::rand({seq_len, batch_size, embed_dim});
        }
        
        // Create key_padding_mask (optional)
        torch::Tensor key_padding_mask;
        bool use_key_padding_mask = offset < Size && Data[offset++] % 2 == 0;
        if (use_key_padding_mask && offset < Size) {
            key_padding_mask = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Create attn_mask (optional)
        torch::Tensor attn_mask;
        bool use_attn_mask = offset < Size && Data[offset++] % 2 == 0;
        if (use_attn_mask && offset < Size) {
            attn_mask = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Forward pass
        torch::Tensor output;
        torch::Tensor attn_output_weights;
        
        if (use_key_padding_mask && use_attn_mask) {
            auto result = mha->forward(query, key, value, key_padding_mask, true, attn_mask);
            output = std::get<0>(result);
            attn_output_weights = std::get<1>(result);
        } else if (use_key_padding_mask) {
            auto result = mha->forward(query, key, value, key_padding_mask, true);
            output = std::get<0>(result);
            attn_output_weights = std::get<1>(result);
        } else if (use_attn_mask) {
            auto result = mha->forward(query, key, value, {}, true, attn_mask);
            output = std::get<0>(result);
            attn_output_weights = std::get<1>(result);
        } else {
            auto result = mha->forward(query, key, value);
            output = std::get<0>(result);
            attn_output_weights = std::get<1>(result);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}