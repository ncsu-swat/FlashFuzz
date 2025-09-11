#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Parse parameters for MultiheadAttention
        int64_t embed_dim = (Data[offset] % 16 + 1) * 8; // Ensure it's a multiple of 8 and non-zero
        offset++;
        
        int64_t num_heads = (Data[offset] % 8) + 1; // Between 1 and 8 heads
        offset++;
        
        double dropout = static_cast<double>(Data[offset]) / 255.0; // Between 0 and 1
        offset++;
        
        bool bias = Data[offset] % 2 == 1; // True or False
        offset++;
        
        bool add_bias_kv = Data[offset] % 2 == 1; // True or False
        offset++;
        
        bool add_zero_attn = Data[offset] % 2 == 1; // True or False
        offset++;
        
        int64_t kdim = embed_dim;
        if (Data[offset] % 4 == 0) { // 25% chance of different kdim
            kdim = (Data[offset] % 16 + 1) * 8;
        }
        offset++;
        
        int64_t vdim = embed_dim;
        if (Data[offset] % 4 == 0) { // 25% chance of different vdim
            vdim = (Data[offset] % 16 + 1) * 8;
        }
        offset++;
        
        // Create query tensor
        torch::Tensor query;
        if (offset < Size) {
            query = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure query has at least 3 dimensions for MultiheadAttention
            if (query.dim() < 3) {
                int64_t seq_len = 10;
                int64_t batch_size = 2;
                query = torch::randn({seq_len, batch_size, embed_dim});
            } else {
                // Ensure the last dimension is embed_dim
                auto sizes = query.sizes().vec();
                sizes[sizes.size() - 1] = embed_dim;
                query = query.reshape(sizes);
            }
        } else {
            int64_t seq_len = 10;
            int64_t batch_size = 2;
            query = torch::randn({seq_len, batch_size, embed_dim});
        }
        
        // Create key tensor
        torch::Tensor key;
        if (offset < Size) {
            key = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure key has at least 3 dimensions
            if (key.dim() < 3) {
                int64_t seq_len = 10;
                int64_t batch_size = 2;
                key = torch::randn({seq_len, batch_size, kdim});
            } else {
                // Ensure the last dimension is kdim
                auto sizes = key.sizes().vec();
                sizes[sizes.size() - 1] = kdim;
                key = key.reshape(sizes);
            }
        } else {
            int64_t seq_len = 10;
            int64_t batch_size = 2;
            key = torch::randn({seq_len, batch_size, kdim});
        }
        
        // Create value tensor
        torch::Tensor value;
        if (offset < Size) {
            value = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure value has at least 3 dimensions
            if (value.dim() < 3) {
                int64_t seq_len = 10;
                int64_t batch_size = 2;
                value = torch::randn({seq_len, batch_size, vdim});
            } else {
                // Ensure the last dimension is vdim
                auto sizes = value.sizes().vec();
                sizes[sizes.size() - 1] = vdim;
                value = value.reshape(sizes);
            }
        } else {
            int64_t seq_len = 10;
            int64_t batch_size = 2;
            value = torch::randn({seq_len, batch_size, vdim});
        }
        
        // Create key_padding_mask tensor (optional)
        torch::Tensor key_padding_mask;
        bool use_key_padding_mask = false;
        if (offset < Size && Data[offset] % 2 == 0) {
            use_key_padding_mask = true;
            offset++;
            if (offset < Size) {
                key_padding_mask = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure key_padding_mask has correct shape (batch_size, seq_len)
                if (key.dim() >= 3) {
                    int64_t batch_size = key.size(1);
                    int64_t seq_len = key.size(0);
                    
                    if (key_padding_mask.dim() != 2 || 
                        key_padding_mask.size(0) != batch_size || 
                        key_padding_mask.size(1) != seq_len) {
                        key_padding_mask = torch::zeros({batch_size, seq_len}, torch::kBool);
                    }
                } else {
                    key_padding_mask = torch::zeros({2, 10}, torch::kBool);
                }
            } else {
                key_padding_mask = torch::zeros({2, 10}, torch::kBool);
            }
        }
        
        // Create attn_mask tensor (optional)
        torch::Tensor attn_mask;
        bool use_attn_mask = false;
        if (offset < Size && Data[offset] % 2 == 0) {
            use_attn_mask = true;
            offset++;
            if (offset < Size) {
                attn_mask = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure attn_mask has correct shape (seq_len, seq_len) or (batch_size * num_heads, seq_len, seq_len)
                if (query.dim() >= 3) {
                    int64_t seq_len = query.size(0);
                    int64_t batch_size = query.size(1);
                    
                    if (attn_mask.dim() != 2 || 
                        attn_mask.size(0) != seq_len || 
                        attn_mask.size(1) != seq_len) {
                        attn_mask = torch::zeros({seq_len, seq_len}, torch::kFloat);
                    }
                } else {
                    attn_mask = torch::zeros({10, 10}, torch::kFloat);
                }
            } else {
                attn_mask = torch::zeros({10, 10}, torch::kFloat);
            }
        }
        
        // Create a MultiheadAttention module
        torch::nn::MultiheadAttention mha(
            torch::nn::MultiheadAttentionOptions(embed_dim, num_heads)
                .dropout(dropout)
                .bias(bias)
                .add_bias_kv(add_bias_kv)
                .add_zero_attn(add_zero_attn)
                .kdim(kdim)
                .vdim(vdim)
        );
        
        // Forward pass with need_weights=false
        torch::Tensor output;
        torch::Tensor attn_weights;
        if (use_key_padding_mask && use_attn_mask) {
            std::tie(output, attn_weights) = mha->forward(query, key, value, key_padding_mask, true, false, attn_mask, false);
        } else if (use_key_padding_mask) {
            std::tie(output, attn_weights) = mha->forward(query, key, value, key_padding_mask, true, false, torch::Tensor{}, false);
        } else if (use_attn_mask) {
            std::tie(output, attn_weights) = mha->forward(query, key, value, torch::Tensor{}, true, false, attn_mask, false);
        } else {
            std::tie(output, attn_weights) = mha->forward(query, key, value, torch::Tensor{}, true, false, torch::Tensor{}, false);
        }
        
        // Forward pass with need_weights=true
        torch::Tensor output2;
        torch::Tensor attn_weights2;
        if (use_key_padding_mask && use_attn_mask) {
            std::tie(output2, attn_weights2) = mha->forward(query, key, value, key_padding_mask, true, false, attn_mask, true);
        } else if (use_key_padding_mask) {
            std::tie(output2, attn_weights2) = mha->forward(query, key, value, key_padding_mask, true, false, torch::Tensor{}, true);
        } else if (use_attn_mask) {
            std::tie(output2, attn_weights2) = mha->forward(query, key, value, torch::Tensor{}, true, false, attn_mask, true);
        } else {
            std::tie(output2, attn_weights2) = mha->forward(query, key, value, torch::Tensor{}, true, false, torch::Tensor{}, true);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
