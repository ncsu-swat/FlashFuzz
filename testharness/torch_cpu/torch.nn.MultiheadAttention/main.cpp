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
        // Need enough bytes for parameters
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Parse dimensions from fuzzer data
        int64_t batch_size = static_cast<int64_t>(Data[offset++] % 8) + 1;      // 1-8
        int64_t seq_len_q = static_cast<int64_t>(Data[offset++] % 16) + 1;      // 1-16
        int64_t seq_len_kv = static_cast<int64_t>(Data[offset++] % 16) + 1;     // 1-16
        int64_t num_heads = static_cast<int64_t>(Data[offset++] % 4) + 1;       // 1-4
        int64_t head_dim = static_cast<int64_t>(Data[offset++] % 8) + 1;        // 1-8
        int64_t embed_dim = num_heads * head_dim;  // Ensure embed_dim is divisible by num_heads

        // Parse boolean options
        bool add_bias_kv = (offset < Size) && (Data[offset++] % 2 == 0);
        bool add_zero_attn = (offset < Size) && (Data[offset++] % 2 == 0);
        bool use_masks = (offset < Size) && (Data[offset++] % 2 == 0);

        // Create MultiheadAttention module
        auto options = torch::nn::MultiheadAttentionOptions(embed_dim, num_heads)
            .bias(true)
            .add_bias_kv(add_bias_kv)
            .add_zero_attn(add_zero_attn)
            .dropout(0.0);  // Use 0 dropout for deterministic fuzzing

        torch::nn::MultiheadAttention mha(options);

        // Create properly shaped tensors for attention
        // MultiheadAttention expects (L, N, E) format by default (batch_first=false)
        torch::Tensor query = torch::randn({seq_len_q, batch_size, embed_dim});
        torch::Tensor key = torch::randn({seq_len_kv, batch_size, embed_dim});
        torch::Tensor value = torch::randn({seq_len_kv, batch_size, embed_dim});

        // Use remaining fuzzer data to perturb tensor values
        if (offset < Size) {
            float scale = static_cast<float>(Data[offset++]) / 25.5f - 5.0f;
            query = query * scale;
        }
        if (offset < Size) {
            float scale = static_cast<float>(Data[offset++]) / 25.5f - 5.0f;
            key = key * scale;
        }
        if (offset < Size) {
            float scale = static_cast<float>(Data[offset++]) / 25.5f - 5.0f;
            value = value * scale;
        }

        // Create optional masks
        torch::Tensor key_padding_mask;
        torch::Tensor attn_mask;

        if (use_masks) {
            // key_padding_mask: (N, S) where S is source sequence length
            key_padding_mask = torch::zeros({batch_size, seq_len_kv}, torch::kBool);
            // Randomly mask some positions based on fuzzer data
            if (offset < Size) {
                int mask_count = Data[offset++] % (seq_len_kv + 1);
                for (int i = 0; i < mask_count && i < seq_len_kv; i++) {
                    int batch_idx = (offset < Size) ? (Data[offset++] % batch_size) : 0;
                    key_padding_mask[batch_idx][i] = true;
                }
            }

            // attn_mask: (L, S) where L is target seq len, S is source seq len
            // Using additive mask (float type with -inf for masked positions)
            attn_mask = torch::zeros({seq_len_q, seq_len_kv});
            if (offset < Size && Data[offset++] % 2 == 0) {
                // Create causal mask
                attn_mask = torch::triu(
                    torch::full({seq_len_q, seq_len_kv}, -std::numeric_limits<float>::infinity()),
                    /*diagonal=*/1
                );
            }
        }

        // Apply MultiheadAttention with need_weights=true
        try {
            auto result = mha->forward(
                query, key, value,
                key_padding_mask.defined() ? key_padding_mask : torch::Tensor(),
                /*need_weights=*/true,
                attn_mask.defined() ? attn_mask : torch::Tensor()
            );

            auto attn_output = std::get<0>(result);
            auto attn_weights = std::get<1>(result);

            // Verify output shapes
            if (attn_output.dim() != 3) {
                return 0;
            }

            // Force computation
            auto sum = attn_output.sum();
            if (attn_weights.defined()) {
                sum = sum + attn_weights.sum();
            }
            (void)sum.item<float>();

        } catch (const c10::Error& e) {
            // Expected PyTorch errors (shape mismatches, etc.)
            return 0;
        }

        // Test with need_weights=false
        try {
            auto result2 = mha->forward(
                query, key, value,
                torch::Tensor(),
                /*need_weights=*/false,
                torch::Tensor()
            );

            auto attn_output2 = std::get<0>(result2);
            (void)attn_output2.sum().item<float>();

        } catch (const c10::Error& e) {
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