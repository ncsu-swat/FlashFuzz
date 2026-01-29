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

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 16) {
            return 0;
        }
        
        // Extract parameters first to ensure d_model is divisible by nhead
        uint8_t nhead_byte = Data[offset++];
        int64_t nhead = (nhead_byte % 4) + 1; // 1-4 heads
        
        uint8_t d_model_mult = Data[offset++];
        int64_t d_model = nhead * ((d_model_mult % 8) + 4); // Ensures d_model is divisible by nhead, range 4*nhead to 11*nhead
        
        uint8_t num_layers_byte = Data[offset++];
        int64_t num_encoder_layers = (num_layers_byte % 2) + 1; // 1-2 layers for performance
        
        uint8_t dim_ff_mult = Data[offset++];
        int64_t dim_feedforward = d_model * ((dim_ff_mult % 3) + 1); // 1x to 3x d_model
        
        uint8_t dropout_byte = Data[offset++];
        double dropout = static_cast<double>(dropout_byte % 50) / 100.0; // 0.0-0.49
        
        uint8_t activation_byte = Data[offset++];
        
        uint8_t flags_byte = Data[offset++];
        bool use_mask = (flags_byte & 0x04) != 0;
        bool use_key_padding_mask = (flags_byte & 0x08) != 0;
        
        // Parse batch and sequence dimensions
        uint8_t batch_byte = Data[offset++];
        int64_t batch_size = (batch_byte % 4) + 1; // 1-4
        
        uint8_t seq_byte = Data[offset++];
        int64_t seq_len = (seq_byte % 8) + 2; // 2-9
        
        // Create source tensor with shape (seq_len, batch_size, d_model)
        // Note: C++ TransformerEncoder expects (seq_len, batch_size, d_model) format
        torch::Tensor src = torch::randn({seq_len, batch_size, d_model});
        
        // Create encoder layer options
        auto encoder_layer_options = torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
            .dim_feedforward(dim_feedforward)
            .dropout(dropout);
        
        // Set activation using the appropriate method
        if (activation_byte % 2 == 1) {
            encoder_layer_options.activation(torch::kGELU);
        } else {
            encoder_layer_options.activation(torch::kReLU);
        }
        
        // Create encoder layer
        auto encoder_layer = torch::nn::TransformerEncoderLayer(encoder_layer_options);
        
        // Create transformer encoder options
        auto encoder_options = torch::nn::TransformerEncoderOptions(encoder_layer, num_encoder_layers);
        
        // Create transformer encoder
        auto transformer_encoder = torch::nn::TransformerEncoder(encoder_options);
        transformer_encoder->eval(); // Set to eval mode to disable dropout randomness
        
        // Create src_mask (attention mask) - shape (seq_len, seq_len)
        torch::Tensor src_mask;
        if (use_mask) {
            // Create a causal mask or random mask
            if (offset < Size && Data[offset++] % 2 == 0) {
                // Causal mask: upper triangular with -inf
                src_mask = torch::zeros({seq_len, seq_len});
                src_mask = src_mask.masked_fill(
                    torch::triu(torch::ones({seq_len, seq_len}), 1).to(torch::kBool),
                    -std::numeric_limits<float>::infinity());
            } else {
                // Zero mask (no masking effect)
                src_mask = torch::zeros({seq_len, seq_len});
            }
        }
        
        // Create src_key_padding_mask - shape (batch_size, seq_len)
        torch::Tensor src_key_padding_mask;
        if (use_key_padding_mask) {
            // Boolean tensor where True means the position should be ignored
            src_key_padding_mask = torch::zeros({batch_size, seq_len}, torch::kBool);
            // Optionally mask some positions based on fuzzer data
            if (offset < Size) {
                int positions_to_mask = Data[offset++] % (seq_len / 2 + 1);
                for (int i = 0; i < positions_to_mask && i < seq_len; i++) {
                    // Mask last positions (like padding)
                    for (int b = 0; b < batch_size; b++) {
                        src_key_padding_mask[b][seq_len - 1 - i] = true;
                    }
                }
            }
        }
        
        // Forward pass with appropriate arguments
        torch::Tensor output;
        try {
            if (use_mask && use_key_padding_mask) {
                output = transformer_encoder->forward(src, src_mask, src_key_padding_mask);
            } else if (use_mask) {
                output = transformer_encoder->forward(src, src_mask);
            } else if (use_key_padding_mask) {
                output = transformer_encoder->forward(src, {}, src_key_padding_mask);
            } else {
                output = transformer_encoder->forward(src);
            }
        } catch (const c10::Error&) {
            // Shape mismatch or other tensor errors - silently continue
            return 0;
        }
        
        // Validate output
        if (output.numel() > 0) {
            auto sum = output.sum().item<float>();
            if (std::isnan(sum) || std::isinf(sum)) {
                return 0;
            }
        }
        
        // Additional coverage: test with different layer configurations
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                // Create encoder with different activation
                auto encoder_layer_options2 = torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
                    .dim_feedforward(dim_feedforward)
                    .dropout(0.0);  // No dropout for this test
                
                // Use opposite activation
                if (activation_byte % 2 == 0) {
                    encoder_layer_options2.activation(torch::kGELU);
                } else {
                    encoder_layer_options2.activation(torch::kReLU);
                }
                
                auto encoder_layer2 = torch::nn::TransformerEncoderLayer(encoder_layer_options2);
                auto encoder_options2 = torch::nn::TransformerEncoderOptions(encoder_layer2, 1);
                auto transformer_encoder2 = torch::nn::TransformerEncoder(encoder_options2);
                transformer_encoder2->eval();
                auto output2 = transformer_encoder2->forward(src);
                (void)output2;
            } catch (const c10::Error&) {
                // Silently handle errors
            }
        }
        
        // Additional coverage: test with norm layer
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                auto encoder_layer3 = torch::nn::TransformerEncoderLayer(encoder_layer_options);
                auto norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}));
                auto encoder_options3 = torch::nn::TransformerEncoderOptions(encoder_layer3, num_encoder_layers)
                    .norm(norm);
                auto transformer_encoder3 = torch::nn::TransformerEncoder(encoder_options3);
                transformer_encoder3->eval();
                auto output3 = transformer_encoder3->forward(src);
                (void)output3;
            } catch (const c10::Error&) {
                // Silently handle errors
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