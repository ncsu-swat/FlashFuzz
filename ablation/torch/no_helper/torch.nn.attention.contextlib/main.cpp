#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <vector>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t*& data, size_t& size, T& out) {
    if (size < sizeof(T)) return false;
    std::memcpy(&out, data, sizeof(T));
    data += sizeof(T);
    size -= sizeof(T);
    return true;
}

// Create tensor from fuzzer input
torch::Tensor createTensorFromBytes(const uint8_t*& data, size_t& size) {
    if (size < 16) {
        return torch::randn({1, 1});
    }
    
    // Consume dimensions
    uint8_t num_dims;
    if (!consumeBytes(data, size, num_dims)) return torch::randn({1});
    num_dims = (num_dims % 4) + 1; // 1-4 dimensions
    
    std::vector<int64_t> shape;
    for (int i = 0; i < num_dims; ++i) {
        uint8_t dim_size;
        if (!consumeBytes(data, size, dim_size)) break;
        shape.push_back((dim_size % 32) + 1); // 1-32 per dimension
    }
    
    if (shape.empty()) shape = {1};
    
    // Consume dtype selector
    uint8_t dtype_selector;
    if (!consumeBytes(data, size, dtype_selector)) {
        return torch::randn(shape);
    }
    
    // Select dtype based on fuzzer input
    torch::ScalarType dtype;
    switch (dtype_selector % 4) {
        case 0: dtype = torch::kFloat32; break;
        case 1: dtype = torch::kFloat64; break;
        case 2: dtype = torch::kFloat16; break;
        default: dtype = torch::kFloat32; break;
    }
    
    // Create tensor with random values
    auto options = torch::TensorOptions().dtype(dtype);
    torch::Tensor tensor = torch::randn(shape, options);
    
    // Optionally make it require grad
    uint8_t requires_grad;
    if (consumeBytes(data, size, requires_grad) && (requires_grad % 2)) {
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            tensor.requires_grad_(true);
        }
    }
    
    return tensor;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        if (size < 20) return 0; // Need minimum bytes for configuration
        
        const uint8_t* ptr = data;
        size_t remaining = size;
        
        // Consume configuration parameters
        uint16_t embed_dim_raw;
        uint8_t num_heads_raw;
        uint8_t dropout_raw;
        uint8_t bias_raw;
        uint8_t add_bias_kv_raw;
        uint8_t add_zero_attn_raw;
        uint8_t batch_first_raw;
        
        if (!consumeBytes(ptr, remaining, embed_dim_raw)) return 0;
        if (!consumeBytes(ptr, remaining, num_heads_raw)) return 0;
        if (!consumeBytes(ptr, remaining, dropout_raw)) return 0;
        if (!consumeBytes(ptr, remaining, bias_raw)) return 0;
        if (!consumeBytes(ptr, remaining, add_bias_kv_raw)) return 0;
        if (!consumeBytes(ptr, remaining, add_zero_attn_raw)) return 0;
        if (!consumeBytes(ptr, remaining, batch_first_raw)) return 0;
        
        // Constrain parameters to reasonable ranges
        int64_t embed_dim = ((embed_dim_raw % 512) + 1) * 8; // 8 to 4096, divisible by 8
        int64_t num_heads = (num_heads_raw % 8) + 1; // 1 to 8
        
        // Ensure embed_dim is divisible by num_heads
        if (embed_dim % num_heads != 0) {
            embed_dim = ((embed_dim / num_heads) + 1) * num_heads;
        }
        
        double dropout = (dropout_raw % 100) / 100.0; // 0.0 to 0.99
        bool bias = bias_raw % 2;
        bool add_bias_kv = add_bias_kv_raw % 2;
        bool add_zero_attn = add_zero_attn_raw % 2;
        bool batch_first = batch_first_raw % 2;
        
        // Create MultiheadAttention module
        auto attention_options = torch::nn::MultiheadAttentionOptions(embed_dim, num_heads)
            .dropout(dropout)
            .bias(bias)
            .add_bias_kv(add_bias_kv)
            .add_zero_attn(add_zero_attn)
            .batch_first(batch_first);
        
        torch::nn::MultiheadAttention attention(attention_options);
        
        // Create input tensors
        torch::Tensor query = createTensorFromBytes(ptr, remaining);
        torch::Tensor key = createTensorFromBytes(ptr, remaining);
        torch::Tensor value = createTensorFromBytes(ptr, remaining);
        
        // Adjust tensor dimensions for attention
        int64_t seq_len = (query.size(0) % 64) + 1;
        int64_t batch_size = (query.size(0) % 16) + 1;
        
        if (batch_first) {
            query = torch::randn({batch_size, seq_len, embed_dim});
            key = torch::randn({batch_size, seq_len, embed_dim});
            value = torch::randn({batch_size, seq_len, embed_dim});
        } else {
            query = torch::randn({seq_len, batch_size, embed_dim});
            key = torch::randn({seq_len, batch_size, embed_dim});
            value = torch::randn({seq_len, batch_size, embed_dim});
        }
        
        // Optionally create attention mask
        torch::Tensor attn_mask;
        uint8_t use_mask;
        if (consumeBytes(ptr, remaining, use_mask) && (use_mask % 3 == 0)) {
            attn_mask = torch::randn({seq_len, seq_len});
        }
        
        // Optionally create key padding mask
        torch::Tensor key_padding_mask;
        uint8_t use_key_mask;
        if (consumeBytes(ptr, remaining, use_key_mask) && (use_key_mask % 3 == 0)) {
            key_padding_mask = torch::randint(0, 2, {batch_size, seq_len}, torch::kBool);
        }
        
        // Forward pass
        try {
            auto [output, attn_weights] = attention->forward(
                query, key, value,
                key_padding_mask,
                /*need_weights=*/true,
                attn_mask
            );
            
            // Additional operations to increase coverage
            if (output.requires_grad()) {
                auto loss = output.sum();
                loss.backward();
            }
            
            // Test with different configurations
            attention->eval();
            auto [eval_output, eval_weights] = attention->forward(
                query, key, value,
                key_padding_mask,
                /*need_weights=*/false,
                attn_mask
            );
            
            // Test training mode
            attention->train();
            auto [train_output, train_weights] = attention->forward(
                query, key, value,
                torch::Tensor(),  // No key padding mask
                /*need_weights=*/true,
                torch::Tensor()   // No attention mask
            );
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid configurations
            return 0;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cout << "Exception caught: unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}