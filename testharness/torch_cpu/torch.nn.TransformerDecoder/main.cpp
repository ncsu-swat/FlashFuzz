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
        
        // Create input tensors for TransformerDecoder
        torch::Tensor tgt;
        torch::Tensor memory;
        
        // Parse tgt tensor
        if (offset < Size) {
            tgt = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Parse memory tensor
        if (offset < Size) {
            memory = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Extract configuration parameters from remaining data
        int d_model = 64;
        int nhead = 8;
        int num_decoder_layers = 2;
        int dim_feedforward = 512;
        float dropout = 0.1;
        
        if (offset + 5 <= Size) {
            d_model = 16 + (Data[offset++] % 112); // Range: 16-128
            nhead = 1 + (Data[offset++] % 8);      // Range: 1-8
            num_decoder_layers = 1 + (Data[offset++] % 3); // Range: 1-3
            dim_feedforward = 32 + (Data[offset++] % 480); // Range: 32-512
            dropout = static_cast<float>(Data[offset++]) / 255.0f; // Range: 0.0-1.0
        }
        
        // Create TransformerDecoderLayer first
        torch::nn::TransformerDecoderLayerOptions layer_options(d_model, nhead);
        layer_options.dim_feedforward(dim_feedforward).dropout(dropout);
        auto decoder_layer = torch::nn::TransformerDecoderLayer(layer_options);
        
        // Create TransformerDecoder with the layer and number of layers
        torch::nn::TransformerDecoderOptions decoder_options(decoder_layer, num_decoder_layers);
        auto decoder = torch::nn::TransformerDecoder(decoder_options);
        
        // Create tgt_mask (optional)
        torch::Tensor tgt_mask;
        if (tgt.dim() >= 2 && offset < Size) {
            bool use_mask = Data[offset++] % 2;
            if (use_mask) {
                int tgt_len = tgt.size(0);
                tgt_mask = torch::ones({tgt_len, tgt_len}, torch::kFloat32);
                if (offset < Size) {
                    bool upper_triangular = Data[offset++] % 2;
                    if (upper_triangular) {
                        tgt_mask = torch::triu(tgt_mask);
                    } else {
                        tgt_mask = torch::tril(tgt_mask);
                    }
                }
            }
        }
        
        // Create memory_mask (optional)
        torch::Tensor memory_mask;
        if (tgt.dim() >= 2 && memory.dim() >= 2 && offset < Size) {
            bool use_memory_mask = Data[offset++] % 2;
            if (use_memory_mask) {
                int tgt_len = tgt.size(0);
                int memory_len = memory.size(0);
                memory_mask = torch::ones({tgt_len, memory_len}, torch::kFloat32);
            }
        }
        
        // Create tgt_key_padding_mask (optional)
        torch::Tensor tgt_key_padding_mask;
        if (tgt.dim() >= 2 && offset < Size) {
            bool use_tgt_padding = Data[offset++] % 2;
            if (use_tgt_padding && tgt.size(1) > 0) {
                int batch_size = tgt.size(1);
                int tgt_len = tgt.size(0);
                tgt_key_padding_mask = torch::zeros({batch_size, tgt_len}, torch::kBool);
                
                // Randomly set some positions to true (masked)
                if (offset < Size) {
                    int mask_density = Data[offset++] % 100;
                    for (int i = 0; i < batch_size; i++) {
                        for (int j = 0; j < tgt_len; j++) {
                            if ((i + j) % 100 < mask_density) {
                                tgt_key_padding_mask.index_put_({i, j}, true);
                            }
                        }
                    }
                }
            }
        }
        
        // Create memory_key_padding_mask (optional)
        torch::Tensor memory_key_padding_mask;
        if (memory.dim() >= 2 && offset < Size) {
            bool use_memory_padding = Data[offset++] % 2;
            if (use_memory_padding && memory.size(1) > 0) {
                int batch_size = memory.size(1);
                int memory_len = memory.size(0);
                memory_key_padding_mask = torch::zeros({batch_size, memory_len}, torch::kBool);
                
                // Randomly set some positions to true (masked)
                if (offset < Size) {
                    int mask_density = Data[offset++] % 100;
                    for (int i = 0; i < batch_size; i++) {
                        for (int j = 0; j < memory_len; j++) {
                            if ((i + j) % 100 < mask_density) {
                                memory_key_padding_mask.index_put_({i, j}, true);
                            }
                        }
                    }
                }
            }
        }
        
        // Apply the TransformerDecoder
        torch::Tensor output = decoder->forward(
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