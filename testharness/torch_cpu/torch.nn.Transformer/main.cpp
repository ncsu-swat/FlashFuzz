#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    if (Size < 16) {
        return 0;
    }

    try
    {
        size_t offset = 0;

        // Extract parameters for the Transformer first
        uint8_t d_model_byte = Data[offset++];
        uint8_t nhead_byte = Data[offset++];
        uint8_t num_encoder_layers_byte = Data[offset++];
        uint8_t num_decoder_layers_byte = Data[offset++];
        uint8_t dim_feedforward_byte = Data[offset++];
        uint8_t dropout_byte = Data[offset++];
        uint8_t src_seq_len_byte = Data[offset++];
        uint8_t tgt_seq_len_byte = Data[offset++];
        uint8_t batch_size_byte = Data[offset++];
        uint8_t mask_flags = Data[offset++];

        // Ensure parameters are within reasonable ranges
        int64_t nhead = 1 + (nhead_byte % 4);  // 1-4 heads
        int64_t d_model = nhead * (2 + (d_model_byte % 8));  // Must be divisible by nhead
        int64_t num_encoder_layers = 1 + (num_encoder_layers_byte % 2);
        int64_t num_decoder_layers = 1 + (num_decoder_layers_byte % 2);
        int64_t dim_feedforward = d_model + (dim_feedforward_byte % 32);
        double dropout = static_cast<double>(dropout_byte % 50) / 100.0;  // 0-0.5

        int64_t src_seq_len = 2 + (src_seq_len_byte % 8);
        int64_t tgt_seq_len = 2 + (tgt_seq_len_byte % 8);
        int64_t batch_size = 1 + (batch_size_byte % 3);

        // Create the Transformer model
        torch::nn::TransformerOptions options(d_model, nhead);
        options.num_encoder_layers(num_encoder_layers);
        options.num_decoder_layers(num_decoder_layers);
        options.dim_feedforward(dim_feedforward);
        options.dropout(dropout);

        auto transformer = torch::nn::Transformer(options);
        transformer->eval();  // Set to eval mode to disable dropout randomness

        // Create input tensors with proper shape [seq_len, batch_size, d_model]
        torch::Tensor src = torch::randn({src_seq_len, batch_size, d_model});
        torch::Tensor tgt = torch::randn({tgt_seq_len, batch_size, d_model});

        // Use remaining fuzzer data to perturb the tensors
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t src_elements = std::min(remaining / 2, static_cast<size_t>(src.numel()));
            auto src_accessor = src.accessor<float, 3>();
            for (size_t i = 0; i < src_elements && offset < Size; i++) {
                int64_t idx = i % src.numel();
                int64_t s = idx / (batch_size * d_model);
                int64_t b = (idx / d_model) % batch_size;
                int64_t d = idx % d_model;
                src_accessor[s][b][d] = static_cast<float>(Data[offset++] - 128) / 64.0f;
            }
        }

        // Create masks based on flags
        torch::Tensor src_mask;
        torch::Tensor tgt_mask;
        torch::Tensor memory_mask;
        torch::Tensor src_key_padding_mask;
        torch::Tensor tgt_key_padding_mask;
        torch::Tensor memory_key_padding_mask;

        if (mask_flags & 0x01) {
            // Square attention mask for source
            src_mask = torch::zeros({src_seq_len, src_seq_len});
        }

        if (mask_flags & 0x02) {
            // Causal mask for target (lower triangular)
            tgt_mask = transformer->generate_square_subsequent_mask(tgt_seq_len);
        }

        if (mask_flags & 0x04) {
            // Memory mask
            memory_mask = torch::zeros({tgt_seq_len, src_seq_len});
        }

        if (mask_flags & 0x08) {
            // Source key padding mask
            src_key_padding_mask = torch::zeros({batch_size, src_seq_len}, torch::kBool);
        }

        if (mask_flags & 0x10) {
            // Target key padding mask
            tgt_key_padding_mask = torch::zeros({batch_size, tgt_seq_len}, torch::kBool);
        }

        if (mask_flags & 0x20) {
            // Memory key padding mask
            memory_key_padding_mask = torch::zeros({batch_size, src_seq_len}, torch::kBool);
        }

        // Apply the transformer
        torch::Tensor output;
        try {
            output = transformer->forward(
                src, tgt, src_mask, tgt_mask, memory_mask,
                src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask
            );
        } catch (const c10::Error &e) {
            // Shape mismatches or other tensor errors - silently ignore
            return 0;
        }

        // Verify output shape matches expected [tgt_seq_len, batch_size, d_model]
        if (output.dim() != 3 ||
            output.size(0) != tgt_seq_len ||
            output.size(1) != batch_size ||
            output.size(2) != d_model) {
            std::cerr << "Unexpected output shape" << std::endl;
        }

        // Access values to ensure computation completed
        if (output.numel() > 0) {
            auto sum = output.sum().item<float>();
            (void)sum;
            
            // Check for NaN/Inf
            if (!std::isfinite(sum)) {
                // This is expected for some inputs, not a bug
            }
        }

        // Test with different activation functions based on mask_flags
        if (mask_flags & 0x40) {
            try {
                torch::nn::TransformerOptions options2(d_model, nhead);
                options2.num_encoder_layers(1);
                options2.num_decoder_layers(1);
                options2.activation(torch::kGELU);
                
                auto transformer2 = torch::nn::Transformer(options2);
                transformer2->eval();
                
                auto output2 = transformer2->forward(src, tgt);
                (void)output2.sum().item<float>();
            } catch (const c10::Error &e) {
                // Ignore errors from this variant
            }
        }

        // Test with batch_first=true option
        if (mask_flags & 0x80) {
            try {
                torch::nn::TransformerOptions options3(d_model, nhead);
                options3.num_encoder_layers(1);
                options3.num_decoder_layers(1);
                options3.batch_first(true);
                
                auto transformer3 = torch::nn::Transformer(options3);
                transformer3->eval();
                
                // With batch_first, shape is [batch_size, seq_len, d_model]
                torch::Tensor src_bf = torch::randn({batch_size, src_seq_len, d_model});
                torch::Tensor tgt_bf = torch::randn({batch_size, tgt_seq_len, d_model});
                
                auto output3 = transformer3->forward(src_bf, tgt_bf);
                (void)output3.sum().item<float>();
            } catch (const c10::Error &e) {
                // Ignore errors from this variant
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