#include "fuzzer_utils.h"
#include <iostream>
#include <ATen/native/transformers/sdp_utils_cpp.h>

// --- Fuzzer Entry Point ---
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
        if (Size < 1) {
            return 0;
        }

        size_t offset = 0;

        // SDPBackend is an enum with values:
        // ERROR = -1, MATH = 0, FLASH_ATTENTION = 1, EFFICIENT_ATTENTION = 2, CUDNN_ATTENTION = 3
        // Select backend based on fuzzer input
        int backend_selector = Data[offset++] % 5;
        at::SDPBackend backend;
        
        switch (backend_selector) {
            case 0:
                backend = at::SDPBackend::math;
                break;
            case 1:
                backend = at::SDPBackend::flash_attention;
                break;
            case 2:
                backend = at::SDPBackend::efficient_attention;
                break;
            case 3:
                backend = at::SDPBackend::cudnn_attention;
                break;
            default:
                backend = at::SDPBackend::error;
                break;
        }

        // Verify the enum value is valid by checking it can be used
        // The backend enum is primarily used for backend selection context
        
        // Create tensors to test with scaled_dot_product_attention
        // which uses SDPBackend internally for backend selection
        if (offset >= Size) {
            return 0;
        }

        // Create proper shaped tensors for attention
        // Query, Key, Value need shape: (batch, num_heads, seq_len, head_dim)
        // or (batch, seq_len, embed_dim) for the simpler interface
        
        int64_t batch_size = 1 + (Data[offset] % 4);
        offset++;
        if (offset >= Size) return 0;
        
        int64_t seq_len = 1 + (Data[offset] % 16);
        offset++;
        if (offset >= Size) return 0;
        
        int64_t embed_dim = 8 * (1 + (Data[offset] % 8)); // Multiple of 8 for efficiency
        offset++;
        
        // Create query, key, value tensors with proper shapes
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        
        torch::Tensor query = torch::randn({batch_size, seq_len, embed_dim}, options);
        torch::Tensor key = torch::randn({batch_size, seq_len, embed_dim}, options);
        torch::Tensor value = torch::randn({batch_size, seq_len, embed_dim}, options);
        
        // Reshape for multi-head attention format: (batch, num_heads, seq_len, head_dim)
        int64_t num_heads = 1;
        if (embed_dim >= 8) {
            num_heads = embed_dim / 8;
        }
        int64_t head_dim = embed_dim / num_heads;
        
        query = query.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
        key = key.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
        value = value.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
        
        // Parse additional parameters from fuzzer data
        double dropout_p = 0.0;
        bool is_causal = false;
        
        if (offset < Size) {
            is_causal = (Data[offset++] & 0x01) != 0;
        }
        
        if (offset < Size) {
            dropout_p = static_cast<double>(Data[offset++] % 10) / 100.0; // 0.0 to 0.09
        }

        // Test scaled_dot_product_attention which internally uses SDPBackend
        // The backend selection happens automatically based on input properties
        try {
            torch::Tensor output = torch::scaled_dot_product_attention(
                query, key, value, 
                /*attn_mask=*/{}, 
                /*dropout_p=*/dropout_p, 
                /*is_causal=*/is_causal,
                /*scale=*/c10::nullopt
            );
            
            // Verify output shape
            if (output.dim() != 4) {
                std::cerr << "Unexpected output dimensions" << std::endl;
            }
        } catch (const c10::Error& e) {
            // Expected exceptions for unsupported configurations
        }

        // Test with causal mask
        try {
            torch::Tensor output = torch::scaled_dot_product_attention(
                query, key, value,
                /*attn_mask=*/{},
                /*dropout_p=*/0.0,
                /*is_causal=*/true,
                /*scale=*/c10::nullopt
            );
        } catch (const c10::Error& e) {
            // Expected exceptions
        }

        // Test with explicit attention mask
        try {
            torch::Tensor attn_mask = torch::zeros({seq_len, seq_len}, options);
            torch::Tensor output = torch::scaled_dot_product_attention(
                query, key, value,
                /*attn_mask=*/attn_mask,
                /*dropout_p=*/0.0,
                /*is_causal=*/false,
                /*scale=*/c10::nullopt
            );
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}