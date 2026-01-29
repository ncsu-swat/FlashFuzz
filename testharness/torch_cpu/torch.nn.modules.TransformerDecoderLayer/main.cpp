#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 20) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract configuration parameters from fuzzer data
        int64_t d_model = 64;
        int64_t nhead = 8;
        int64_t dim_feedforward = 256;
        double dropout = 0.0; // Use 0 dropout for deterministic behavior
        
        if (offset + 4 <= Size) {
            uint32_t config;
            std::memcpy(&config, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            
            // d_model must be divisible by nhead
            nhead = (config & 0x07) + 1; // 1-8 heads
            d_model = nhead * (((config >> 3) & 0x0F) + 2) * 4; // Multiple of nhead, at least 8*nhead
        }
        
        if (offset + 4 <= Size) {
            uint32_t config;
            std::memcpy(&config, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            dim_feedforward = (config % 512) + 32;
        }
        
        // Extract sequence lengths and batch size
        int64_t tgt_seq_len = 4;
        int64_t memory_seq_len = 6;
        int64_t batch_size = 2;
        
        if (offset + 3 <= Size) {
            tgt_seq_len = (Data[offset++] % 8) + 1;
            memory_seq_len = (Data[offset++] % 8) + 1;
            batch_size = (Data[offset++] % 4) + 1;
        }
        
        // Create properly shaped tensors for TransformerDecoderLayer
        // tgt: (tgt_seq_len, batch_size, d_model)
        // memory: (memory_seq_len, batch_size, d_model)
        torch::Tensor tgt = torch::randn({tgt_seq_len, batch_size, d_model});
        torch::Tensor memory = torch::randn({memory_seq_len, batch_size, d_model});
        
        // Use remaining fuzzer data to perturb tensor values
        if (offset + 8 <= Size) {
            float scale;
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(scale) && std::abs(scale) < 100.0f) {
                tgt = tgt * scale;
            }
        }
        
        // Select activation function type
        bool use_gelu = false;
        if (offset < Size) {
            use_gelu = (Data[offset++] % 2) == 1;
        }
        
        // Create TransformerDecoderLayer options
        torch::nn::TransformerDecoderLayerOptions options(d_model, nhead);
        options.dim_feedforward(dim_feedforward)
               .dropout(dropout);
        
        // Set activation - the C++ API uses activation() with kReLU or kGELU enum
        if (use_gelu) {
            options.activation(torch::kGELU);
        } else {
            options.activation(torch::kReLU);
        }
        
        torch::nn::TransformerDecoderLayer decoder_layer(options);
        
        // Set to eval mode to avoid randomness from dropout
        decoder_layer->eval();
        
        // Determine mask configuration
        uint8_t mask_config = 0;
        if (offset < Size) {
            mask_config = Data[offset++];
        }
        
        bool use_tgt_mask = (mask_config & 0x01) != 0;
        bool use_memory_mask = (mask_config & 0x02) != 0;
        bool use_tgt_key_padding_mask = (mask_config & 0x04) != 0;
        bool use_memory_key_padding_mask = (mask_config & 0x08) != 0;
        
        // Create masks with proper shapes
        torch::Tensor tgt_mask;
        torch::Tensor memory_mask;
        torch::Tensor tgt_key_padding_mask;
        torch::Tensor memory_key_padding_mask;
        
        if (use_tgt_mask) {
            // tgt_mask: (tgt_seq_len, tgt_seq_len) - additive mask
            tgt_mask = torch::zeros({tgt_seq_len, tgt_seq_len});
            // Create causal mask
            for (int64_t i = 0; i < tgt_seq_len; i++) {
                for (int64_t j = i + 1; j < tgt_seq_len; j++) {
                    tgt_mask[i][j] = -1e9f;
                }
            }
        }
        
        if (use_memory_mask) {
            // memory_mask: (tgt_seq_len, memory_seq_len)
            memory_mask = torch::zeros({tgt_seq_len, memory_seq_len});
        }
        
        if (use_tgt_key_padding_mask) {
            // tgt_key_padding_mask: (batch_size, tgt_seq_len)
            tgt_key_padding_mask = torch::zeros({batch_size, tgt_seq_len}, torch::kBool);
            // Randomly mask some positions but ensure at least one is unmasked per sequence
            for (int64_t b = 0; b < batch_size && offset < Size; b++) {
                for (int64_t s = 0; s < tgt_seq_len - 1 && offset < Size; s++) {
                    tgt_key_padding_mask[b][s] = (Data[offset++] % 4) == 0; // 25% masked
                }
            }
        }
        
        if (use_memory_key_padding_mask) {
            // memory_key_padding_mask: (batch_size, memory_seq_len)
            memory_key_padding_mask = torch::zeros({batch_size, memory_seq_len}, torch::kBool);
            for (int64_t b = 0; b < batch_size && offset < Size; b++) {
                for (int64_t s = 0; s < memory_seq_len - 1 && offset < Size; s++) {
                    memory_key_padding_mask[b][s] = (Data[offset++] % 4) == 0;
                }
            }
        }
        
        // Forward pass
        torch::Tensor output;
        try {
            output = decoder_layer->forward(
                tgt, 
                memory, 
                tgt_mask, 
                memory_mask, 
                tgt_key_padding_mask, 
                memory_key_padding_mask
            );
        } catch (const c10::Error &e) {
            // Shape mismatches or other expected errors
            return 0;
        }
        
        // Ensure the output is used to prevent optimization
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        // Test forward without masks as well for coverage
        if ((mask_config & 0x10) != 0) {
            try {
                torch::Tensor output2 = decoder_layer->forward(tgt, memory);
                if (output2.defined()) {
                    volatile float sum2 = output2.sum().item<float>();
                    (void)sum2;
                }
            } catch (const c10::Error &e) {
                // Ignore expected errors
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}