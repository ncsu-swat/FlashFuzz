#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful tensors
        if (Size < 10) {
            return 0;
        }
        
        // Parse configuration parameters from the input data
        uint8_t embed_dim = 0;
        uint8_t num_heads = 0;
        bool bias = false;
        float dropout = 0.0f;
        bool add_bias_kv = false;
        bool add_zero_attn = false;
        
        if (offset < Size) embed_dim = (Data[offset++] % 8 + 1) * 8; // Make embed_dim a multiple of 8
        if (offset < Size) {
            num_heads = Data[offset++] % 8 + 1; // 1-8 heads
            // Ensure embed_dim is divisible by num_heads
            embed_dim = (embed_dim / num_heads) * num_heads;
            if (embed_dim == 0) embed_dim = num_heads;
        }
        if (offset < Size) bias = Data[offset++] % 2 == 0;
        if (offset < Size) {
            uint8_t dropout_byte = Data[offset++];
            dropout = static_cast<float>(dropout_byte) / 255.0f;
        }
        if (offset < Size) add_bias_kv = Data[offset++] % 2 == 0;
        if (offset < Size) add_zero_attn = Data[offset++] % 2 == 0;
        
        // Create MultiheadAttention module
        torch::nn::MultiheadAttention mha(
            torch::nn::MultiheadAttentionOptions(embed_dim, num_heads)
                .bias(bias)
                .dropout(dropout)
                .add_bias_kv(add_bias_kv)
                .add_zero_attn(add_zero_attn)
        );
        
        // Create input tensors
        torch::Tensor query;
        torch::Tensor key;
        torch::Tensor value;
        
        try {
            query = fuzzer_utils::createTensor(Data, Size, offset);
            key = fuzzer_utils::createTensor(Data, Size, offset);
            value = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape tensors to valid dimensions for MultiheadAttention if needed
            // MultiheadAttention expects query, key, value of shape (L, N, E)
            // where L = sequence length, N = batch size, E = embedding dimension
            
            // Ensure tensors have at least 3 dimensions
            if (query.dim() < 3) {
                std::vector<int64_t> new_shape;
                if (query.dim() == 0) {
                    new_shape = {1, 1, embed_dim};
                } else if (query.dim() == 1) {
                    new_shape = {query.size(0), 1, embed_dim};
                } else { // dim == 2
                    new_shape = {query.size(0), query.size(1), embed_dim};
                }
                query = query.reshape(new_shape);
            }
            
            if (key.dim() < 3) {
                std::vector<int64_t> new_shape;
                if (key.dim() == 0) {
                    new_shape = {1, 1, embed_dim};
                } else if (key.dim() == 1) {
                    new_shape = {key.size(0), 1, embed_dim};
                } else { // dim == 2
                    new_shape = {key.size(0), key.size(1), embed_dim};
                }
                key = key.reshape(new_shape);
            }
            
            if (value.dim() < 3) {
                std::vector<int64_t> new_shape;
                if (value.dim() == 0) {
                    new_shape = {1, 1, embed_dim};
                } else if (value.dim() == 1) {
                    new_shape = {value.size(0), 1, embed_dim};
                } else { // dim == 2
                    new_shape = {value.size(0), value.size(1), embed_dim};
                }
                value = value.reshape(new_shape);
            }
            
            // Ensure the last dimension is embed_dim
            if (query.size(2) != embed_dim) {
                query = query.reshape({query.size(0), query.size(1), embed_dim});
            }
            if (key.size(2) != embed_dim) {
                key = key.reshape({key.size(0), key.size(1), embed_dim});
            }
            if (value.size(2) != embed_dim) {
                value = value.reshape({value.size(0), value.size(1), embed_dim});
            }
            
            // Create optional parameters
            torch::Tensor key_padding_mask;
            torch::Tensor attn_mask;
            
            bool use_key_padding_mask = false;
            bool use_attn_mask = false;
            
            if (offset < Size) {
                use_key_padding_mask = Data[offset++] % 2 == 0;
            }
            
            if (offset < Size) {
                use_attn_mask = Data[offset++] % 2 == 0;
            }
            
            if (use_key_padding_mask) {
                try {
                    key_padding_mask = fuzzer_utils::createTensor(Data, Size, offset);
                    // Reshape to match batch size and key sequence length
                    if (key_padding_mask.dim() == 0) {
                        key_padding_mask = key_padding_mask.reshape({1, 1});
                    } else if (key_padding_mask.dim() == 1) {
                        key_padding_mask = key_padding_mask.reshape({1, key_padding_mask.size(0)});
                    }
                    
                    // Ensure dimensions match batch size and key sequence length
                    key_padding_mask = key_padding_mask.reshape({key.size(1), key.size(0)});
                    
                    // Convert to boolean mask
                    key_padding_mask = key_padding_mask.to(torch::kBool);
                } catch (const std::exception& e) {
                    use_key_padding_mask = false;
                }
            }
            
            if (use_attn_mask) {
                try {
                    attn_mask = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Reshape to match query and key sequence lengths
                    if (attn_mask.dim() == 0) {
                        attn_mask = attn_mask.reshape({1, 1});
                    } else if (attn_mask.dim() == 1) {
                        attn_mask = attn_mask.reshape({1, attn_mask.size(0)});
                    }
                    
                    // Ensure dimensions match query and key sequence lengths
                    attn_mask = attn_mask.reshape({query.size(0), key.size(0)});
                    
                    // Convert to appropriate dtype (float for additive mask, bool for multiplicative mask)
                    if (offset < Size && Data[offset++] % 2 == 0) {
                        attn_mask = attn_mask.to(torch::kFloat32);
                    } else {
                        attn_mask = attn_mask.to(torch::kBool);
                    }
                } catch (const std::exception& e) {
                    use_attn_mask = false;
                }
            }
            
            // Call the MultiheadAttention forward function
            torch::Tensor attn_output;
            torch::Tensor attn_output_weights;
            
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
            
            // Test some operations on the output to ensure it's valid
            auto output_sum = attn_output.sum();
            auto weights_sum = attn_output_weights.sum();
            
            // Test with different need_weights values
            std::tie(attn_output, attn_output_weights) = mha->forward(
                query, key, value, {}, false, {}
            );
            
            // Test with average_attn_weights=false
            if (offset < Size && Data[offset++] % 2 == 0) {
                std::tie(attn_output, attn_output_weights) = mha->forward(
                    query, key, value, {}, true, {}, false
                );
            }
            
        } catch (const std::exception& e) {
            // Catch exceptions from tensor creation or reshaping
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