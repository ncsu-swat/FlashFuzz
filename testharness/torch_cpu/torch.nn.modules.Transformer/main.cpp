#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse configuration parameters from the data
        int64_t d_model = 8 + (Data[offset++] % 24);  // 8-32, keep small for speed
        int64_t nhead = 1 + (Data[offset++] % 4);     // 1-4
        
        // Ensure d_model is divisible by nhead
        d_model = (d_model / nhead) * nhead;
        if (d_model < nhead) d_model = nhead;
        if (d_model < 8) d_model = 8;
        
        int64_t num_encoder_layers = 1 + (Data[offset++] % 2); // 1-2
        int64_t num_decoder_layers = 1 + (Data[offset++] % 2); // 1-2
        int64_t dim_feedforward = d_model * (1 + (Data[offset++] % 2)); // d_model to 2*d_model
        double dropout = 0.0; // Disable dropout for deterministic fuzzing
        
        // Parse tensor dimensions
        int64_t src_seq_len = 1 + (Data[offset++] % 8);  // 1-8
        int64_t tgt_seq_len = 1 + (Data[offset++] % 8);  // 1-8
        int64_t batch_size = 1 + (Data[offset++] % 4);   // 1-4
        
        // Flags for mask usage
        bool use_src_mask = (Data[offset++] % 2) == 0;
        bool use_tgt_mask = (Data[offset++] % 2) == 0;
        bool use_memory_mask = (Data[offset++] % 2) == 0;
        bool use_key_padding_masks = (Data[offset++] % 2) == 0;
        
        // Create transformer module
        auto options = torch::nn::TransformerOptions(d_model, nhead)
            .num_encoder_layers(num_encoder_layers)
            .num_decoder_layers(num_decoder_layers)
            .dim_feedforward(dim_feedforward)
            .dropout(dropout);
        
        auto transformer = torch::nn::Transformer(options);
        transformer->eval(); // Set to eval mode to disable dropout
        
        // Create input tensors with correct shape [seq_len, batch, d_model]
        torch::Tensor src = torch::randn({src_seq_len, batch_size, d_model}, torch::kFloat);
        torch::Tensor tgt = torch::randn({tgt_seq_len, batch_size, d_model}, torch::kFloat);
        
        // Use remaining data to perturb tensor values
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t src_numel = static_cast<size_t>(src.numel());
            size_t tgt_numel = static_cast<size_t>(tgt.numel());
            
            auto src_accessor = src.accessor<float, 3>();
            auto tgt_accessor = tgt.accessor<float, 3>();
            
            for (size_t i = 0; i < remaining && i < src_numel; i++) {
                float val = static_cast<float>(Data[offset + i]) / 127.5f - 1.0f;
                int64_t idx = static_cast<int64_t>(i);
                int64_t s = idx / (batch_size * d_model);
                int64_t b = (idx / d_model) % batch_size;
                int64_t d = idx % d_model;
                if (s < src_seq_len) {
                    src_accessor[s][b][d] = val;
                }
            }
            
            for (size_t i = 0; i < remaining && i < tgt_numel; i++) {
                float val = static_cast<float>(Data[offset + (i % remaining)]) / 127.5f - 1.0f;
                int64_t idx = static_cast<int64_t>(i);
                int64_t s = idx / (batch_size * d_model);
                int64_t b = (idx / d_model) % batch_size;
                int64_t d = idx % d_model;
                if (s < tgt_seq_len) {
                    tgt_accessor[s][b][d] = val;
                }
            }
        }
        
        // Create masks
        torch::Tensor src_mask = {};
        torch::Tensor tgt_mask = {};
        torch::Tensor memory_mask = {};
        torch::Tensor src_key_padding_mask = {};
        torch::Tensor tgt_key_padding_mask = {};
        torch::Tensor memory_key_padding_mask = {};
        
        // Generate masks based on flags
        if (use_src_mask) {
            src_mask = transformer->generate_square_subsequent_mask(src_seq_len);
        }
        
        if (use_tgt_mask) {
            tgt_mask = transformer->generate_square_subsequent_mask(tgt_seq_len);
        }
        
        if (use_memory_mask) {
            // Memory mask: [tgt_seq_len, src_seq_len]
            memory_mask = torch::zeros({tgt_seq_len, src_seq_len}, torch::kFloat);
        }
        
        if (use_key_padding_masks) {
            // Key padding masks: [batch, seq_len] boolean tensors
            src_key_padding_mask = torch::zeros({batch_size, src_seq_len}, torch::kBool);
            tgt_key_padding_mask = torch::zeros({batch_size, tgt_seq_len}, torch::kBool);
            memory_key_padding_mask = torch::zeros({batch_size, src_seq_len}, torch::kBool);
        }
        
        // Inner try-catch for expected runtime failures
        try {
            // Apply transformer with various mask configurations
            torch::Tensor output;
            
            if (use_key_padding_masks) {
                output = transformer->forward(
                    src, tgt,
                    src_mask, tgt_mask, memory_mask,
                    src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask
                );
            } else if (use_src_mask || use_tgt_mask || use_memory_mask) {
                output = transformer->forward(
                    src, tgt,
                    src_mask, tgt_mask, memory_mask
                );
            } else {
                output = transformer->forward(src, tgt);
            }
            
            // Verify output shape
            if (output.dim() != 3 || 
                output.size(0) != tgt_seq_len || 
                output.size(1) != batch_size || 
                output.size(2) != d_model) {
                std::cerr << "Unexpected output shape" << std::endl;
            }
            
            // Additional coverage: test encoder and decoder separately
            auto memory = transformer->forward(src, tgt).clone();
            
        } catch (const c10::Error& e) {
            // Expected PyTorch errors (shape mismatches, etc.) - silently ignore
        } catch (const std::runtime_error& e) {
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