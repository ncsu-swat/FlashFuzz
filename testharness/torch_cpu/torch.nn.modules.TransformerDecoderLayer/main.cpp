#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensors for TransformerDecoderLayer
        torch::Tensor tgt;
        torch::Tensor memory;
        
        try {
            tgt = fuzzer_utils::createTensor(Data, Size, offset);
            memory = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception &e) {
            return 0;
        }
        
        // Extract configuration parameters from the remaining data
        int64_t d_model = 64;
        int64_t nhead = 8;
        int64_t dim_feedforward = 512;
        double dropout = 0.1;
        
        if (offset + 8 <= Size) {
            int64_t d_model_raw;
            std::memcpy(&d_model_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            d_model = std::abs(d_model_raw) % 256 + 16;
        }
        
        if (offset + 8 <= Size) {
            int64_t nhead_raw;
            std::memcpy(&nhead_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            nhead = std::abs(nhead_raw) % 16 + 1;
            
            // Ensure d_model is divisible by nhead
            d_model = nhead * (d_model / nhead + (d_model % nhead > 0));
        }
        
        if (offset + 8 <= Size) {
            int64_t dim_feedforward_raw;
            std::memcpy(&dim_feedforward_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dim_feedforward = std::abs(dim_feedforward_raw) % 2048 + 32;
        }
        
        if (offset + 8 <= Size) {
            double dropout_raw;
            std::memcpy(&dropout_raw, Data + offset, sizeof(double));
            offset += sizeof(double);
            dropout = std::abs(dropout_raw) / (1.0 + std::abs(dropout_raw)); // Normalize to [0, 1)
        }
        
        // Create activation function
        torch::nn::TransformerDecoderLayerOptions::activation_t activation = torch::kReLU;
        if (offset < Size) {
            uint8_t activation_selector = Data[offset++];
            switch (activation_selector % 2) {
                case 0: activation = torch::kReLU; break;
                case 1: activation = torch::kGELU; break;
            }
        }
        
        // Create layer normalization options
        bool normalize_before = false;
        if (offset < Size) {
            normalize_before = Data[offset++] % 2 == 0;
        }
        
        // Create TransformerDecoderLayer
        torch::nn::TransformerDecoderLayerOptions options(d_model, nhead);
        options.dim_feedforward(dim_feedforward)
               .dropout(dropout)
               .activation(activation)
               .normalize_before(normalize_before);
        
        auto decoder_layer = torch::nn::TransformerDecoderLayer(options);
        
        // Set to eval mode to avoid randomness from dropout
        decoder_layer->eval();
        
        // Apply the decoder layer
        torch::Tensor output;
        
        // Try with different mask configurations
        torch::Tensor tgt_mask;
        torch::Tensor memory_mask;
        torch::Tensor tgt_key_padding_mask;
        torch::Tensor memory_key_padding_mask;
        
        bool use_tgt_mask = false;
        bool use_memory_mask = false;
        bool use_tgt_key_padding_mask = false;
        bool use_memory_key_padding_mask = false;
        
        if (offset < Size) {
            uint8_t mask_config = Data[offset++];
            use_tgt_mask = (mask_config & 0x01) != 0;
            use_memory_mask = (mask_config & 0x02) != 0;
            use_tgt_key_padding_mask = (mask_config & 0x04) != 0;
            use_memory_key_padding_mask = (mask_config & 0x08) != 0;
        }
        
        // Create masks if needed
        if (use_tgt_mask && tgt.dim() >= 2) {
            int64_t seq_len = tgt.size(0);
            tgt_mask = torch::zeros({seq_len, seq_len}, torch::kFloat);
            
            // Create a causal mask (lower triangular)
            for (int64_t i = 0; i < seq_len; i++) {
                for (int64_t j = 0; j <= i; j++) {
                    tgt_mask.index_put_({i, j}, 0.0);
                }
                for (int64_t j = i + 1; j < seq_len; j++) {
                    tgt_mask.index_put_({i, j}, -1e9);
                }
            }
        }
        
        if (use_memory_mask && tgt.dim() >= 2 && memory.dim() >= 2) {
            int64_t tgt_seq_len = tgt.size(0);
            int64_t memory_seq_len = memory.size(0);
            memory_mask = torch::zeros({tgt_seq_len, memory_seq_len}, torch::kFloat);
        }
        
        if (use_tgt_key_padding_mask && tgt.dim() >= 2) {
            int64_t batch_size = tgt.size(1);
            int64_t tgt_seq_len = tgt.size(0);
            tgt_key_padding_mask = torch::zeros({batch_size, tgt_seq_len}, torch::kBool);
            
            // Randomly mask some positions
            if (offset + batch_size * tgt_seq_len <= Size) {
                for (int64_t i = 0; i < batch_size; i++) {
                    for (int64_t j = 0; j < tgt_seq_len; j++) {
                        tgt_key_padding_mask.index_put_({i, j}, Data[offset++] % 2 == 0);
                    }
                }
            }
        }
        
        if (use_memory_key_padding_mask && memory.dim() >= 2) {
            int64_t batch_size = memory.size(1);
            int64_t memory_seq_len = memory.size(0);
            memory_key_padding_mask = torch::zeros({batch_size, memory_seq_len}, torch::kBool);
            
            // Randomly mask some positions
            if (offset + batch_size * memory_seq_len <= Size) {
                for (int64_t i = 0; i < batch_size; i++) {
                    for (int64_t j = 0; j < memory_seq_len; j++) {
                        memory_key_padding_mask.index_put_({i, j}, Data[offset++] % 2 == 0);
                    }
                }
            }
        }
        
        // Forward pass with the created tensors and masks
        output = decoder_layer->forward(
            tgt, 
            memory, 
            tgt_mask, 
            memory_mask, 
            tgt_key_padding_mask, 
            memory_key_padding_mask
        );
        
        // Ensure the output is used to prevent optimization
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}