#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        torch::NoGradGuard no_grad;
        
        size_t offset = 0;
        
        if (Size < 16) {
            return 0;
        }
        
        // Parse configuration parameters from the input data
        int64_t embed_dim = (Data[offset++] % 8 + 1) * 8; // 8, 16, 24, ..., 64
        int64_t num_heads = Data[offset++] % 8 + 1; // 1-8 heads
        
        // Ensure embed_dim is divisible by num_heads
        embed_dim = (embed_dim / num_heads) * num_heads;
        if (embed_dim == 0) embed_dim = num_heads * 8;
        
        bool bias = Data[offset++] % 2 == 0;
        float dropout = static_cast<float>(Data[offset++]) / 255.0f * 0.5f; // Cap at 0.5
        bool add_bias_kv = Data[offset++] % 2 == 0;
        bool add_zero_attn = Data[offset++] % 2 == 0;
        
        // Parse sequence lengths and batch size
        int64_t seq_len_q = (Data[offset++] % 8) + 1;  // 1-8
        int64_t seq_len_kv = (Data[offset++] % 8) + 1; // 1-8
        int64_t batch_size = (Data[offset++] % 4) + 1; // 1-4
        
        // Create MultiheadAttention module
        torch::nn::MultiheadAttention mha(
            torch::nn::MultiheadAttentionOptions(embed_dim, num_heads)
                .bias(bias)
                .dropout(dropout)
                .add_bias_kv(add_bias_kv)
                .add_zero_attn(add_zero_attn)
        );
        mha->eval(); // Set to eval mode to disable dropout
        
        // Create input tensors with proper shapes directly
        // MultiheadAttention expects (L, N, E) format: (seq_len, batch, embed_dim)
        torch::Tensor query = torch::randn({seq_len_q, batch_size, embed_dim});
        torch::Tensor key = torch::randn({seq_len_kv, batch_size, embed_dim});
        torch::Tensor value = torch::randn({seq_len_kv, batch_size, embed_dim});
        
        // Use fuzzer data to modify tensor values
        if (offset + 4 <= Size) {
            float scale_q = static_cast<float>(Data[offset++]) / 128.0f;
            query = query * scale_q;
        }
        if (offset + 4 <= Size) {
            float scale_k = static_cast<float>(Data[offset++]) / 128.0f;
            key = key * scale_k;
        }
        if (offset + 4 <= Size) {
            float scale_v = static_cast<float>(Data[offset++]) / 128.0f;
            value = value * scale_v;
        }
        
        // Decide on optional masks
        bool use_key_padding_mask = (offset < Size) && (Data[offset++] % 2 == 0);
        bool use_attn_mask = (offset < Size) && (Data[offset++] % 2 == 0);
        
        torch::Tensor key_padding_mask;
        torch::Tensor attn_mask;
        
        if (use_key_padding_mask) {
            // key_padding_mask shape: (N, S) where S is source sequence length
            key_padding_mask = torch::zeros({batch_size, seq_len_kv}, torch::kBool);
            // Randomly mask some positions
            if (offset < Size) {
                int num_masked = Data[offset++] % (seq_len_kv + 1);
                for (int i = 0; i < num_masked && offset < Size; i++) {
                    int64_t pos = Data[offset++] % seq_len_kv;
                    int64_t batch_idx = (offset < Size) ? Data[offset++] % batch_size : 0;
                    key_padding_mask[batch_idx][pos] = true;
                }
            }
        }
        
        if (use_attn_mask) {
            // attn_mask shape: (L, S) or (N*num_heads, L, S)
            // Use 2D mask for simplicity
            bool use_float_mask = (offset < Size) && (Data[offset++] % 2 == 0);
            if (use_float_mask) {
                attn_mask = torch::zeros({seq_len_q, seq_len_kv}, torch::kFloat32);
                // Set some positions to -inf for masking
                if (offset < Size) {
                    int num_masked = Data[offset++] % (seq_len_q * seq_len_kv / 2 + 1);
                    for (int i = 0; i < num_masked && offset + 1 < Size; i++) {
                        int64_t row = Data[offset++] % seq_len_q;
                        int64_t col = Data[offset++] % seq_len_kv;
                        attn_mask[row][col] = -std::numeric_limits<float>::infinity();
                    }
                }
            } else {
                attn_mask = torch::zeros({seq_len_q, seq_len_kv}, torch::kBool);
                if (offset < Size) {
                    int num_masked = Data[offset++] % (seq_len_q * seq_len_kv / 2 + 1);
                    for (int i = 0; i < num_masked && offset + 1 < Size; i++) {
                        int64_t row = Data[offset++] % seq_len_q;
                        int64_t col = Data[offset++] % seq_len_kv;
                        attn_mask[row][col] = true;
                    }
                }
            }
        }
        
        // Call the MultiheadAttention forward function
        torch::Tensor attn_output;
        torch::Tensor attn_output_weights;
        
        try {
            if (use_key_padding_mask && use_attn_mask) {
                std::tie(attn_output, attn_output_weights) = mha->forward(
                    query, key, value, key_padding_mask, true, attn_mask
                );
            } else if (use_key_padding_mask) {
                std::tie(attn_output, attn_output_weights) = mha->forward(
                    query, key, value, key_padding_mask, true, {}
                );
            } else if (use_attn_mask) {
                std::tie(attn_output, attn_output_weights) = mha->forward(
                    query, key, value, {}, true, attn_mask
                );
            } else {
                std::tie(attn_output, attn_output_weights) = mha->forward(
                    query, key, value, {}, true, {}
                );
            }
            
            // Verify output shapes and values
            auto output_sum = attn_output.sum();
            auto weights_sum = attn_output_weights.sum();
            
            // Test with need_weights=false
            std::tie(attn_output, attn_output_weights) = mha->forward(
                query, key, value, {}, false, {}
            );
            
            // Test with average_attn_weights=false if supported
            if (offset < Size && Data[offset++] % 2 == 0) {
                try {
                    std::tie(attn_output, attn_output_weights) = mha->forward(
                        query, key, value, {}, true, {}, false
                    );
                } catch (...) {
                    // average_attn_weights parameter might not be supported in all versions
                }
            }
        } catch (const c10::Error& e) {
            // Expected errors from invalid configurations, silently ignore
            return 0;
        } catch (const std::runtime_error& e) {
            // Shape mismatches and similar runtime errors, silently ignore
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}