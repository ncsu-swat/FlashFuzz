#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need sufficient data for parameters
        if (Size < 16) {
            return 0;
        }

        size_t offset = 0;

        // Parse dimensions from fuzzer data
        uint8_t batch_size = (Data[offset++] % 4) + 1;      // 1-4
        uint8_t num_heads = (Data[offset++] % 4) + 1;       // 1-4
        uint8_t seq_len_q = (Data[offset++] % 16) + 1;      // 1-16
        uint8_t seq_len_kv = (Data[offset++] % 16) + 1;     // 1-16
        uint8_t head_dim = (Data[offset++] % 32) + 8;       // 8-39

        // Parse dropout probability
        float dropout_p = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&dropout_p, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Clamp to valid range and avoid NaN
            if (std::isnan(dropout_p) || std::isinf(dropout_p)) {
                dropout_p = 0.0f;
            }
            dropout_p = std::max(0.0f, std::min(1.0f, dropout_p));
        }

        // Parse is_causal flag
        bool is_causal = false;
        if (offset < Size) {
            is_causal = static_cast<bool>(Data[offset++] & 0x01);
        }

        // Parse scale factor
        c10::optional<double> scale = c10::nullopt;
        if (offset < Size && (Data[offset++] & 0x01)) {
            if (offset + sizeof(float) <= Size) {
                float scale_val;
                std::memcpy(&scale_val, Data + offset, sizeof(float));
                offset += sizeof(float);
                if (!std::isnan(scale_val) && !std::isinf(scale_val) && scale_val > 0.0f) {
                    scale = static_cast<double>(scale_val);
                }
            }
        }

        // Create tensors with proper attention shapes: (batch, num_heads, seq_len, head_dim)
        torch::Tensor query = torch::randn({batch_size, num_heads, seq_len_q, head_dim});
        torch::Tensor key = torch::randn({batch_size, num_heads, seq_len_kv, head_dim});
        torch::Tensor value = torch::randn({batch_size, num_heads, seq_len_kv, head_dim});

        // Test 1: Basic call with required parameters only
        try {
            auto output1 = torch::scaled_dot_product_attention(query, key, value);
            (void)output1;
        } catch (const c10::Error& e) {
            // Expected for some input combinations
        }

        // Test 2: Call with attention mask
        try {
            // Attention mask shape: (seq_len_q, seq_len_kv) or broadcastable
            torch::Tensor attn_mask = torch::zeros({seq_len_q, seq_len_kv});
            auto output2 = torch::scaled_dot_product_attention(
                query, key, value, attn_mask);
            (void)output2;
        } catch (const c10::Error& e) {
            // Expected for some input combinations
        }

        // Test 3: Call with dropout (only in training mode)
        try {
            auto output3 = torch::scaled_dot_product_attention(
                query, key, value, 
                /*attn_mask=*/{},
                /*dropout_p=*/dropout_p);
            (void)output3;
        } catch (const c10::Error& e) {
            // Expected for some input combinations
        }

        // Test 4: Call with is_causal flag
        // When is_causal=true, seq_len_q should equal seq_len_kv for standard causal mask
        try {
            if (is_causal && seq_len_q == seq_len_kv) {
                auto output4 = torch::scaled_dot_product_attention(
                    query, key, value,
                    /*attn_mask=*/{},
                    /*dropout_p=*/0.0,
                    /*is_causal=*/true);
                (void)output4;
            }
        } catch (const c10::Error& e) {
            // Expected for some input combinations
        }

        // Test 5: Call with scale parameter
        try {
            auto output5 = torch::scaled_dot_product_attention(
                query, key, value,
                /*attn_mask=*/{},
                /*dropout_p=*/0.0,
                /*is_causal=*/false,
                /*scale=*/scale);
            (void)output5;
        } catch (const c10::Error& e) {
            // Expected for some input combinations
        }

        // Test 6: Different dtypes
        try {
            torch::Tensor query_f64 = query.to(torch::kFloat64);
            torch::Tensor key_f64 = key.to(torch::kFloat64);
            torch::Tensor value_f64 = value.to(torch::kFloat64);
            auto output6 = torch::scaled_dot_product_attention(query_f64, key_f64, value_f64);
            (void)output6;
        } catch (const c10::Error& e) {
            // Expected for some input combinations
        }

    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}