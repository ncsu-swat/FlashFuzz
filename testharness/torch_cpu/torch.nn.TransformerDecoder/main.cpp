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
        
        // Extract configuration parameters first to determine tensor shapes
        int nhead = 1 + (Data[offset++] % 8);      // Range: 1-8
        // d_model must be divisible by nhead
        int d_model_multiplier = 1 + (Data[offset++] % 16); // Range: 1-16
        int d_model = nhead * d_model_multiplier;  // Ensures d_model % nhead == 0
        int num_decoder_layers = 1 + (Data[offset++] % 3); // Range: 1-3
        int dim_feedforward = 32 + (Data[offset++] % 224); // Range: 32-256
        float dropout = static_cast<float>(Data[offset++] % 50) / 100.0f; // Range: 0.0-0.49
        
        // Extract tensor dimensions
        int tgt_seq_len = 1 + (Data[offset++] % 16);    // Range: 1-16
        int memory_seq_len = 1 + (Data[offset++] % 16); // Range: 1-16
        int batch_size = 1 + (Data[offset++] % 8);      // Range: 1-8
        
        // Create input tensors with correct shapes for TransformerDecoder
        // tgt: (tgt_seq_len, batch_size, d_model)
        // memory: (memory_seq_len, batch_size, d_model)
        torch::Tensor tgt = torch::randn({tgt_seq_len, batch_size, d_model});
        torch::Tensor memory = torch::randn({memory_seq_len, batch_size, d_model});
        
        // Create TransformerDecoderLayer first
        torch::nn::TransformerDecoderLayerOptions layer_options(d_model, nhead);
        layer_options.dim_feedforward(dim_feedforward).dropout(dropout);
        auto decoder_layer = torch::nn::TransformerDecoderLayer(layer_options);
        
        // Create TransformerDecoder with the layer and number of layers
        torch::nn::TransformerDecoderOptions decoder_options(decoder_layer, num_decoder_layers);
        auto decoder = torch::nn::TransformerDecoder(decoder_options);
        
        // Set to eval mode to disable dropout for deterministic behavior
        decoder->eval();
        
        // Create optional masks based on remaining data
        torch::Tensor tgt_mask;
        torch::Tensor memory_mask;
        torch::Tensor tgt_key_padding_mask;
        torch::Tensor memory_key_padding_mask;
        
        if (offset < Size) {
            uint8_t mask_flags = Data[offset++];
            
            // tgt_mask: (tgt_seq_len, tgt_seq_len)
            if (mask_flags & 0x01) {
                tgt_mask = torch::zeros({tgt_seq_len, tgt_seq_len});
                if (mask_flags & 0x02) {
                    // Upper triangular mask (causal)
                    tgt_mask = torch::triu(torch::ones({tgt_seq_len, tgt_seq_len}) * (-1e9), 1);
                }
            }
            
            // memory_mask: (tgt_seq_len, memory_seq_len)
            if (mask_flags & 0x04) {
                memory_mask = torch::zeros({tgt_seq_len, memory_seq_len});
            }
            
            // tgt_key_padding_mask: (batch_size, tgt_seq_len)
            if (mask_flags & 0x08) {
                tgt_key_padding_mask = torch::zeros({batch_size, tgt_seq_len}, torch::kBool);
                // Set some padding positions if we have more data
                if (offset < Size) {
                    int num_padded = Data[offset++] % (tgt_seq_len + 1);
                    for (int b = 0; b < batch_size; b++) {
                        for (int i = tgt_seq_len - num_padded; i < tgt_seq_len; i++) {
                            if (i >= 0) {
                                tgt_key_padding_mask.index_put_({b, i}, true);
                            }
                        }
                    }
                }
            }
            
            // memory_key_padding_mask: (batch_size, memory_seq_len)
            if (mask_flags & 0x10) {
                memory_key_padding_mask = torch::zeros({batch_size, memory_seq_len}, torch::kBool);
                if (offset < Size) {
                    int num_padded = Data[offset++] % (memory_seq_len + 1);
                    for (int b = 0; b < batch_size; b++) {
                        for (int i = memory_seq_len - num_padded; i < memory_seq_len; i++) {
                            if (i >= 0) {
                                memory_key_padding_mask.index_put_({b, i}, true);
                            }
                        }
                    }
                }
            }
        }
        
        // Apply the TransformerDecoder
        torch::Tensor output;
        try {
            output = decoder->forward(
                tgt, 
                memory, 
                tgt_mask, 
                memory_mask, 
                tgt_key_padding_mask, 
                memory_key_padding_mask
            );
        } catch (const c10::Error&) {
            // Shape mismatches or other tensor errors are expected with some inputs
            return 0;
        }
        
        // Ensure the output is used to prevent optimization
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}