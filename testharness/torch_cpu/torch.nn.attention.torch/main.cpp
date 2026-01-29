#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need enough data for parameters
        if (Size < 16) {
            return 0;
        }

        size_t offset = 0;

        // Parse dimensions from fuzzer data
        uint8_t batch_size = std::max(1, static_cast<int>(Data[offset++] % 4 + 1));      // 1-4
        uint8_t num_heads = std::max(1, static_cast<int>(Data[offset++] % 4 + 1));       // 1-4
        uint8_t seq_len_q = std::max(1, static_cast<int>(Data[offset++] % 16 + 1));      // 1-16
        uint8_t seq_len_kv = std::max(1, static_cast<int>(Data[offset++] % 16 + 1));     // 1-16
        uint8_t head_dim = std::max(1, static_cast<int>(Data[offset++] % 32 + 8));       // 8-39

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
        std::optional<double> scale = std::nullopt;
        if (offset < Size && (Data[offset++] & 0x01)) {
            if (offset + sizeof(float) <= Size) {
                float temp_scale;
                std::memcpy(&temp_scale, Data + offset, sizeof(float));
                offset += sizeof(float);
                // Avoid invalid scale values
                if (!std::isnan(temp_scale) && !std::isinf(temp_scale) && temp_scale != 0.0f) {
                    scale = static_cast<double>(temp_scale);
                }
            }
        }

        // Parse whether to use attention mask
        bool use_attn_mask = false;
        if (offset < Size) {
            use_attn_mask = static_cast<bool>(Data[offset++] & 0x01);
        }

        // Create properly shaped tensors for scaled_dot_product_attention
        // Shape: (batch_size, num_heads, seq_len, head_dim)
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        
        torch::Tensor query = torch::randn({batch_size, num_heads, seq_len_q, head_dim}, options);
        torch::Tensor key = torch::randn({batch_size, num_heads, seq_len_kv, head_dim}, options);
        torch::Tensor value = torch::randn({batch_size, num_heads, seq_len_kv, head_dim}, options);

        // Create attention mask if requested
        std::optional<torch::Tensor> attn_mask_opt = std::nullopt;
        if (use_attn_mask && !is_causal) {
            // Attention mask shape: (seq_len_q, seq_len_kv) or (batch, num_heads, seq_len_q, seq_len_kv)
            if (offset < Size && (Data[offset++] & 0x01)) {
                // 4D mask
                attn_mask_opt = torch::zeros({batch_size, num_heads, seq_len_q, seq_len_kv}, options);
            } else {
                // 2D mask
                attn_mask_opt = torch::zeros({seq_len_q, seq_len_kv}, options);
            }
        }

        // Test 1: Basic call with required parameters only
        try {
            auto result1 = torch::scaled_dot_product_attention(query, key, value);
            (void)result1;
        } catch (const c10::Error& e) {
            // Expected for some input combinations
        }

        // Test 2: Call with attention mask (no dropout during inference)
        if (attn_mask_opt.has_value()) {
            try {
                auto result2 = torch::scaled_dot_product_attention(
                    query, key, value, attn_mask_opt, 0.0, false, scale);
                (void)result2;
            } catch (const c10::Error& e) {
                // Expected for some input combinations
            }
        }

        // Test 3: Call with is_causal flag
        if (is_causal && seq_len_q == seq_len_kv) {
            try {
                auto result3 = torch::scaled_dot_product_attention(
                    query, key, value, std::nullopt, 0.0, true, scale);
                (void)result3;
            } catch (const c10::Error& e) {
                // Expected for some input combinations
            }
        }

        // Test 4: Call with scale parameter
        if (scale.has_value()) {
            try {
                auto result4 = torch::scaled_dot_product_attention(
                    query, key, value, std::nullopt, 0.0, false, scale);
                (void)result4;
            } catch (const c10::Error& e) {
                // Expected for some input combinations
            }
        }

        // Test 5: Full parameter call (dropout=0 for deterministic testing)
        try {
            auto result5 = torch::scaled_dot_product_attention(
                query, key, value, attn_mask_opt, 0.0, is_causal && seq_len_q == seq_len_kv, scale);
            (void)result5;
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