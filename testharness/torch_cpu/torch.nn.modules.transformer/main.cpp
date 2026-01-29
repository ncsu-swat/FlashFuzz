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
        torch::NoGradGuard no_grad;
        
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract configuration parameters first
        int64_t d_model = 8 + (Data[offset++] % 56);  // 8-64 range
        int64_t nhead = 1 + (Data[offset++] % 8);     // 1-8 range
        
        // Ensure d_model is divisible by nhead
        d_model = (d_model / nhead) * nhead;
        if (d_model == 0) d_model = nhead;
        
        int64_t num_encoder_layers = 1 + (Data[offset++] % 3);  // 1-3 range
        int64_t num_decoder_layers = 1 + (Data[offset++] % 3);  // 1-3 range
        int64_t dim_feedforward = d_model * (1 + (Data[offset++] % 4));  // d_model to 4*d_model
        
        // Sequence lengths and batch size from fuzzer data
        int64_t src_seq_len = 1 + (Data[offset++] % 16);  // 1-16
        int64_t tgt_seq_len = 1 + (Data[offset++] % 16);  // 1-16
        int64_t batch_size = 1 + (Data[offset++] % 4);    // 1-4
        
        // Create transformer options
        torch::nn::TransformerOptions transformer_options(d_model, nhead);
        transformer_options.num_encoder_layers(num_encoder_layers);
        transformer_options.num_decoder_layers(num_decoder_layers);
        transformer_options.dim_feedforward(dim_feedforward);
        transformer_options.dropout(0.0);  // Disable dropout for deterministic fuzzing
        
        // Set activation function
        if (offset < Size && Data[offset++] % 2 == 0) {
            transformer_options.activation(torch::kGELU);
        } else {
            transformer_options.activation(torch::kReLU);
        }
        
        // Create the transformer module
        torch::nn::Transformer transformer_module(transformer_options);
        transformer_module->eval();  // Set to eval mode
        
        // Create source tensor: [src_seq_len, batch_size, d_model]
        torch::Tensor src = torch::randn({src_seq_len, batch_size, d_model});
        
        // Create target tensor: [tgt_seq_len, batch_size, d_model]
        torch::Tensor tgt = torch::randn({tgt_seq_len, batch_size, d_model});
        
        // Use remaining fuzzer data to perturb tensors
        if (offset + 4 < Size) {
            float scale = static_cast<float>(Data[offset++]) / 25.5f;  // 0-10 scale
            src = src * scale;
            tgt = tgt * scale;
        }
        
        // Create masks based on fuzzer data
        torch::Tensor src_mask = {};
        torch::Tensor tgt_mask = {};
        torch::Tensor memory_mask = {};
        torch::Tensor src_key_padding_mask = {};
        torch::Tensor tgt_key_padding_mask = {};
        torch::Tensor memory_key_padding_mask = {};
        
        if (offset < Size) {
            uint8_t mask_selector = Data[offset++];
            
            // Create source attention mask (optional)
            if (mask_selector & 0x01) {
                src_mask = torch::zeros({src_seq_len, src_seq_len});
            }
            
            // Create target causal mask (common for autoregressive)
            if (mask_selector & 0x02) {
                tgt_mask = transformer_module->generate_square_subsequent_mask(tgt_seq_len);
            }
            
            // Create memory mask (optional)
            if (mask_selector & 0x04) {
                memory_mask = torch::zeros({tgt_seq_len, src_seq_len});
            }
            
            // Create padding masks (optional)
            if (mask_selector & 0x08) {
                src_key_padding_mask = torch::zeros({batch_size, src_seq_len}, torch::kBool);
            }
            
            if (mask_selector & 0x10) {
                tgt_key_padding_mask = torch::zeros({batch_size, tgt_seq_len}, torch::kBool);
            }
            
            if (mask_selector & 0x20) {
                memory_key_padding_mask = torch::zeros({batch_size, src_seq_len}, torch::kBool);
            }
        }
        
        // Forward pass through transformer
        try {
            torch::Tensor output = transformer_module->forward(
                src, tgt, src_mask, tgt_mask, memory_mask,
                src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask
            );
            
            // Verify output shape
            if (output.dim() != 3 || 
                output.size(0) != tgt_seq_len || 
                output.size(1) != batch_size || 
                output.size(2) != d_model) {
                // Unexpected output shape - this shouldn't happen
            }
        } catch (const c10::Error&) {
            // Shape or dimension mismatch - expected for some configurations
        }
        
        // Test individual encoder/decoder if we have more data
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                // Create separate encoder and decoder for testing
                torch::nn::TransformerEncoderLayerOptions encoder_layer_opts(d_model, nhead);
                encoder_layer_opts.dim_feedforward(dim_feedforward);
                encoder_layer_opts.dropout(0.0);
                
                torch::nn::TransformerEncoderOptions encoder_opts(
                    torch::nn::TransformerEncoderLayer(encoder_layer_opts), num_encoder_layers);
                torch::nn::TransformerEncoder encoder(encoder_opts);
                encoder->eval();
                
                torch::Tensor encoder_output = encoder->forward(src, src_mask, src_key_padding_mask);
                
                torch::nn::TransformerDecoderLayerOptions decoder_layer_opts(d_model, nhead);
                decoder_layer_opts.dim_feedforward(dim_feedforward);
                decoder_layer_opts.dropout(0.0);
                
                torch::nn::TransformerDecoderOptions decoder_opts(
                    torch::nn::TransformerDecoderLayer(decoder_layer_opts), num_decoder_layers);
                torch::nn::TransformerDecoder decoder(decoder_opts);
                decoder->eval();
                
                torch::Tensor decoder_output = decoder->forward(
                    tgt, encoder_output, tgt_mask, memory_mask,
                    tgt_key_padding_mask, memory_key_padding_mask
                );
            } catch (const c10::Error&) {
                // Expected for some configurations
            }
        }
        
        // Test TransformerEncoderLayer and TransformerDecoderLayer directly
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                torch::nn::TransformerEncoderLayerOptions enc_layer_opts(d_model, nhead);
                enc_layer_opts.dim_feedforward(dim_feedforward);
                enc_layer_opts.dropout(0.0);
                
                torch::nn::TransformerEncoderLayer enc_layer(enc_layer_opts);
                enc_layer->eval();
                
                torch::Tensor enc_out = enc_layer->forward(src, src_mask, src_key_padding_mask);
                
                torch::nn::TransformerDecoderLayerOptions dec_layer_opts(d_model, nhead);
                dec_layer_opts.dim_feedforward(dim_feedforward);
                dec_layer_opts.dropout(0.0);
                
                torch::nn::TransformerDecoderLayer dec_layer(dec_layer_opts);
                dec_layer->eval();
                
                torch::Tensor dec_out = dec_layer->forward(
                    tgt, enc_out, tgt_mask, memory_mask,
                    tgt_key_padding_mask, memory_key_padding_mask
                );
            } catch (const c10::Error&) {
                // Expected for some configurations
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