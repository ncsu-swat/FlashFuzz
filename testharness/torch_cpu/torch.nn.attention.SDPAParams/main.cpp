#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need enough data for parameters
        if (Size < 16) {
            return 0;
        }
        
        // Parse dimensions for SDPA tensors
        // SDPA expects 4D tensors: (batch, num_heads, seq_len, head_dim)
        int batch_size = (Data[offset++] % 4) + 1;      // 1-4
        int num_heads = (Data[offset++] % 4) + 1;       // 1-4
        int seq_len_q = (Data[offset++] % 16) + 1;      // 1-16
        int seq_len_kv = (Data[offset++] % 16) + 1;     // 1-16
        int head_dim = (Data[offset++] % 32) + 1;       // 1-32
        
        // Create query tensor: (batch, num_heads, seq_len_q, head_dim)
        torch::Tensor query = torch::randn({batch_size, num_heads, seq_len_q, head_dim});
        
        // Create key tensor: (batch, num_heads, seq_len_kv, head_dim)
        torch::Tensor key = torch::randn({batch_size, num_heads, seq_len_kv, head_dim});
        
        // Create value tensor: (batch, num_heads, seq_len_kv, head_dim)
        torch::Tensor value = torch::randn({batch_size, num_heads, seq_len_kv, head_dim});
        
        // Optionally use fuzzer data to influence tensor values
        if (offset < Size) {
            float scale_factor = static_cast<float>(Data[offset++]) / 255.0f * 2.0f;
            query = query * scale_factor;
        }
        
        // Create optional attn_mask tensor
        c10::optional<torch::Tensor> attn_mask = c10::nullopt;
        bool use_mask = false;
        
        if (offset < Size && Data[offset++] % 3 == 0) {
            use_mask = true;
            // Attention mask shape: (seq_len_q, seq_len_kv) or broadcastable
            attn_mask = torch::randn({seq_len_q, seq_len_kv});
        }
        
        // Parse dropout probability
        double dropout_p = 0.0;
        if (offset < Size) {
            // Map to 0.0-0.5 range (high dropout can cause issues)
            dropout_p = static_cast<double>(Data[offset++]) / 255.0 * 0.5;
        }
        
        // Parse is_causal flag
        bool is_causal = false;
        if (offset < Size) {
            is_causal = (Data[offset++] % 2 == 0);
            // is_causal requires seq_len_q <= seq_len_kv and no attn_mask
            if (is_causal) {
                use_mask = false;
                attn_mask = c10::nullopt;
            }
        }
        
        // Parse scale factor
        c10::optional<double> scale = c10::nullopt;
        if (offset < Size && Data[offset++] % 2 == 0) {
            if (offset < Size) {
                // Scale typically around 1/sqrt(head_dim), allow some variation
                float temp_scale = static_cast<float>(Data[offset++]) / 255.0f * 2.0f + 0.01f;
                scale = static_cast<double>(temp_scale);
            }
        }
        
        // Apply the scaled_dot_product_attention operation
        torch::Tensor output;
        
        try {
            if (use_mask && attn_mask.has_value()) {
                output = torch::scaled_dot_product_attention(
                    query, key, value, attn_mask.value(), dropout_p, is_causal, scale);
            } else {
                output = torch::scaled_dot_product_attention(
                    query, key, value, {}, dropout_p, is_causal, scale);
            }
            
            // Perform operations on output to ensure computation happens
            auto sum = output.sum();
            auto mean = output.mean();
            
            // Verify output shape
            auto out_sizes = output.sizes();
            if (out_sizes.size() != 4) {
                std::cerr << "Unexpected output dimensions" << std::endl;
            }
            
            // Prevent compiler optimization
            if (sum.item<float>() == -12345.6789f) {
                std::cerr << "Unlikely sum value" << std::endl;
            }
        }
        catch (const c10::Error &e) {
            // Expected errors from incompatible inputs - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}