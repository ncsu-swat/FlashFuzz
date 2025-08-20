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
        int64_t embed_dim = (Data[0] % 8 + 1) * 8;  // Make divisible by 8 for quantization
        int64_t num_heads = Data[1] % 8 + 1;
        bool bias = Data[2] % 2;
        float dropout_p = static_cast<float>(Data[3]) / 255.0f;
        bool add_bias_kv = Data[4] % 2;
        bool add_zero_attn = Data[5] % 2;
        int64_t kdim = (Data[6] % 2) ? embed_dim : (Data[6] % 8 + 1) * 8;
        int64_t vdim = (Data[7] % 2) ? embed_dim : (Data[7] % 8 + 1) * 8;
        
        offset = 8;
        
        // Create query tensor
        int64_t seq_len = Data[offset++] % 10 + 1;
        int64_t batch_size = Data[offset++] % 5 + 1;
        
        // Create tensors with appropriate shapes for MultiheadAttention
        torch::Tensor query;
        torch::Tensor key;
        torch::Tensor value;
        
        try {
            // Create query tensor with shape [seq_len, batch_size, embed_dim]
            std::vector<int64_t> query_shape = {seq_len, batch_size, embed_dim};
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            query = torch::rand(query_shape, options);
            
            // Create key and value tensors with same shape as query or different kdim/vdim
            std::vector<int64_t> key_shape = {seq_len, batch_size, kdim};
            std::vector<int64_t> value_shape = {seq_len, batch_size, vdim};
            key = torch::rand(key_shape, options);
            value = torch::rand(value_shape, options);
            
            // Create key padding mask (optional)
            torch::Tensor key_padding_mask;
            if (offset < Size && Data[offset++] % 2) {
                key_padding_mask = torch::randint(0, 2, {batch_size, seq_len}, torch::kBool);
            }
            
            // Create attention mask (optional)
            torch::Tensor attn_mask;
            if (offset < Size && Data[offset++] % 2) {
                if (offset < Size && Data[offset++] % 2) {
                    // 2D attention mask
                    attn_mask = torch::randint(0, 2, {seq_len, seq_len}, torch::kBool);
                } else {
                    // 3D attention mask
                    attn_mask = torch::randint(0, 2, {batch_size * num_heads, seq_len, seq_len}, torch::kBool);
                }
            }
            
            // Quantize the input tensors
            auto scale = 1.0f / 128.0f;
            auto zero_point = 0;
            
            query = torch::quantize_per_tensor(query, scale, zero_point, torch::kQUInt8);
            key = torch::quantize_per_tensor(key, scale, zero_point, torch::kQUInt8);
            value = torch::quantize_per_tensor(value, scale, zero_point, torch::kQUInt8);
            
            // Create the MultiheadAttention module
            torch::nn::MultiheadAttention mha(
                torch::nn::MultiheadAttentionOptions(embed_dim, num_heads)
                    .dropout(dropout_p)
                    .bias(bias)
                    .add_bias_kv(add_bias_kv)
                    .add_zero_attn(add_zero_attn)
                    .kdim(kdim)
                    .vdim(vdim)
            );
            
            // Apply the MultiheadAttention directly (quantization API not available in C++)
            torch::Tensor output;
            torch::Tensor attn_output_weights;
            
            if (key_padding_mask.defined() && attn_mask.defined()) {
                std::tie(output, attn_output_weights) = mha->forward(query, key, value, key_padding_mask, true, attn_mask);
            } else if (key_padding_mask.defined()) {
                std::tie(output, attn_output_weights) = mha->forward(query, key, value, key_padding_mask, true);
            } else if (attn_mask.defined()) {
                std::tie(output, attn_output_weights) = mha->forward(query, key, value, torch::Tensor{}, true, attn_mask);
            } else {
                std::tie(output, attn_output_weights) = mha->forward(query, key, value);
            }
            
            // Verify output shape
            auto expected_output_shape = query.sizes();
            if (output.sizes() != expected_output_shape) {
                throw std::runtime_error("Output shape mismatch");
            }
            
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and can be safely ignored
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