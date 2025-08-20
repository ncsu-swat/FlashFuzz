#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Parse parameters for MultiheadAttention
        int64_t embed_dim = (Data[0] % 16 + 1) * 8;  // Ensure it's a multiple of 8 and > 0
        int64_t num_heads = Data[1] % 8 + 1;         // Between 1 and 8 heads
        double dropout = static_cast<double>(Data[2]) / 255.0;  // Between 0 and 1
        bool bias = Data[3] % 2 == 0;                // Random boolean
        bool add_bias_kv = Data[4] % 2 == 0;         // Random boolean
        bool add_zero_attn = Data[5] % 2 == 0;       // Random boolean
        int64_t kdim = (Data[6] % 2 == 0) ? embed_dim : (Data[6] % 16 + 1) * 8;  // Either embed_dim or different
        int64_t vdim = (Data[7] % 2 == 0) ? embed_dim : (Data[7] % 16 + 1) * 8;  // Either embed_dim or different
        
        offset = 8;
        
        // Create MultiheadAttention module
        torch::nn::MultiheadAttention mha(
            torch::nn::MultiheadAttentionOptions(embed_dim, num_heads)
                .dropout(dropout)
                .bias(bias)
                .add_bias_kv(add_bias_kv)
                .add_zero_attn(add_zero_attn)
                .kdim(kdim)
                .vdim(vdim)
        );
        
        // Create input tensors
        torch::Tensor query;
        torch::Tensor key;
        torch::Tensor value;
        
        try {
            query = fuzzer_utils::createTensor(Data, Size, offset);
            key = (offset < Size) ? fuzzer_utils::createTensor(Data, Size, offset) : query.clone();
            value = (offset < Size) ? fuzzer_utils::createTensor(Data, Size, offset) : key.clone();
        } catch (const std::exception& e) {
            // If tensor creation fails, create some default tensors
            int64_t seq_len = 10;
            int64_t batch_size = 2;
            
            query = torch::rand({seq_len, batch_size, embed_dim});
            key = torch::rand({seq_len, batch_size, kdim});
            value = torch::rand({seq_len, batch_size, vdim});
        }
        
        // Ensure tensors have compatible shapes for MHA
        // MHA expects [seq_len, batch_size, embed_dim] for query
        // and [seq_len, batch_size, kdim/vdim] for key/value
        if (query.dim() < 2) {
            query = query.unsqueeze(0).unsqueeze(0);
            if (query.dim() < 3) {
                query = query.expand({1, 1, embed_dim});
            }
        }
        
        if (key.dim() < 2) {
            key = key.unsqueeze(0).unsqueeze(0);
            if (key.dim() < 3) {
                key = key.expand({1, 1, kdim});
            }
        }
        
        if (value.dim() < 2) {
            value = value.unsqueeze(0).unsqueeze(0);
            if (value.dim() < 3) {
                value = value.expand({1, 1, vdim});
            }
        }
        
        // Reshape tensors if needed to match expected dimensions
        if (query.dim() > 0 && query.size(query.dim()-1) != embed_dim) {
            query = query.reshape({-1, embed_dim}).unsqueeze(1);
        }
        
        if (key.dim() > 0 && key.size(key.dim()-1) != kdim) {
            key = key.reshape({-1, kdim}).unsqueeze(1);
        }
        
        if (value.dim() > 0 && value.size(value.dim()-1) != vdim) {
            value = value.reshape({-1, vdim}).unsqueeze(1);
        }
        
        // Create optional parameters
        torch::Tensor key_padding_mask;
        torch::Tensor attn_mask;
        
        if (offset + 1 < Size && Data[offset] % 3 == 0) {
            // Create key_padding_mask (batch_size x seq_len)
            try {
                key_padding_mask = fuzzer_utils::createTensor(Data, Size, offset);
                if (key_padding_mask.dim() > 0 && key.dim() > 1) {
                    // Reshape to match batch_size x seq_len
                    key_padding_mask = key_padding_mask.reshape({key.size(1), key.size(0)}).to(torch::kBool);
                }
            } catch (...) {
                // Skip if creation fails
            }
        }
        
        if (offset + 1 < Size && Data[offset] % 3 == 1) {
            // Create attention mask
            try {
                attn_mask = fuzzer_utils::createTensor(Data, Size, offset);
                if (attn_mask.dim() > 0 && query.dim() > 0 && key.dim() > 0) {
                    // Reshape to match seq_len_q x seq_len_k
                    attn_mask = attn_mask.reshape({query.size(0), key.size(0)});
                }
            } catch (...) {
                // Skip if creation fails
            }
        }
        
        // Call the MultiheadAttention forward function
        auto result = mha->forward(query, key, value, key_padding_mask, false, attn_mask);
        
        // Access the output to ensure computation is not optimized away
        auto output = std::get<0>(result);
        auto attention_weights = std::get<1>(result);
        
        // Perform some operation on the output to ensure it's used
        auto sum = output.sum();
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}