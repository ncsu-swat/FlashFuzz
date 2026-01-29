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
        if (Size < 12) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse input parameters for TransformerDecoderLayer
        // nhead must divide d_model evenly
        int64_t nhead = (Data[offset++] % 4) + 1;  // 1, 2, 3, or 4
        int64_t d_model = nhead * ((Data[offset++] % 8) + 1);  // Ensures d_model % nhead == 0
        int64_t dim_feedforward = ((Data[offset] << 8) | Data[offset + 1]) % 512 + 64;
        offset += 2;
        double dropout = static_cast<double>(Data[offset++]) / 255.0 * 0.5;  // Cap at 0.5
        
        // Parse sequence lengths
        int64_t tgt_seq_len = (Data[offset++] % 8) + 1;
        int64_t memory_seq_len = (Data[offset++] % 8) + 1;
        int64_t batch_size = (Data[offset++] % 4) + 1;
        
        // Flags for optional masks
        bool use_tgt_mask = Data[offset++] & 1;
        bool use_memory_mask = Data[offset++] & 1;
        bool use_tgt_key_padding_mask = Data[offset++] & 1;
        bool use_memory_key_padding_mask = Data[offset++] & 1;
        
        // Create TransformerDecoderLayer
        // Note: batch_first and norm_first are not available in C++ API
        auto options = torch::nn::TransformerDecoderLayerOptions(d_model, nhead)
            .dim_feedforward(dim_feedforward)
            .dropout(dropout);
        
        torch::nn::TransformerDecoderLayer decoder_layer(options);
        decoder_layer->eval();  // Disable dropout for deterministic behavior
        
        // Create tgt and memory tensors with proper shapes
        // Default is batch_first=false, so shape is (seq_len, batch_size, d_model)
        torch::Tensor tgt = torch::randn({tgt_seq_len, batch_size, d_model});
        torch::Tensor memory = torch::randn({memory_seq_len, batch_size, d_model});
        
        // Create optional mask tensors with proper shapes
        torch::Tensor tgt_mask = {};
        torch::Tensor memory_mask = {};
        torch::Tensor tgt_key_padding_mask = {};
        torch::Tensor memory_key_padding_mask = {};
        
        if (use_tgt_mask) {
            // tgt_mask: (tgt_seq_len, tgt_seq_len) - attention mask
            tgt_mask = torch::zeros({tgt_seq_len, tgt_seq_len});
        }
        
        if (use_memory_mask) {
            // memory_mask: (tgt_seq_len, memory_seq_len)
            memory_mask = torch::zeros({tgt_seq_len, memory_seq_len});
        }
        
        if (use_tgt_key_padding_mask) {
            // tgt_key_padding_mask: (batch_size, tgt_seq_len) - bool tensor
            tgt_key_padding_mask = torch::zeros({batch_size, tgt_seq_len}, torch::kBool);
        }
        
        if (use_memory_key_padding_mask) {
            // memory_key_padding_mask: (batch_size, memory_seq_len) - bool tensor
            memory_key_padding_mask = torch::zeros({batch_size, memory_seq_len}, torch::kBool);
        }
        
        // Inner try-catch for expected runtime errors (shape mismatches, etc.)
        try {
            torch::Tensor output = decoder_layer->forward(
                tgt, 
                memory, 
                tgt_mask, 
                memory_mask, 
                tgt_key_padding_mask, 
                memory_key_padding_mask
            );
            
            // Ensure the output is used to prevent optimization
            if (output.defined()) {
                volatile auto sum = output.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error&) {
            // Expected errors from shape mismatches - silently ignore
        } catch (const std::runtime_error&) {
            // Expected runtime errors - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}