#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    torch::NoGradGuard no_grad;

    if (Size < 16) {
        return 0;
    }

    try
    {
        size_t offset = 0;

        // Extract parameters from fuzzer data
        uint8_t d_model_idx = Data[offset++] % 4;  // 0-3
        uint8_t nhead_idx = Data[offset++] % 4;    // 0-3
        uint8_t dim_ff_idx = Data[offset++] % 3;   // 0-2
        uint8_t dropout_byte = Data[offset++];
        uint8_t activation_idx = Data[offset++] % 2;
        uint8_t seq_len_byte = Data[offset++];
        uint8_t batch_size_byte = Data[offset++];
        uint8_t use_mask = Data[offset++];
        uint8_t use_key_padding_mask = Data[offset++];

        // Map to valid d_model values (must be divisible by nhead)
        // Use fixed valid combinations
        int64_t nhead_options[] = {1, 2, 4, 8};
        int64_t nhead = nhead_options[nhead_idx];
        
        // d_model must be divisible by nhead, use multiples
        int64_t d_model_multipliers[] = {1, 2, 4, 8};
        int64_t d_model = nhead * d_model_multipliers[d_model_idx] * 8; // 8, 16, 32, 64 per head
        
        // Reasonable dim_feedforward values
        int64_t dim_ff_options[] = {64, 128, 256};
        int64_t dim_feedforward = dim_ff_options[dim_ff_idx];
        
        float dropout = static_cast<float>(dropout_byte) / 512.0f; // 0 to ~0.5
        
        // Sequence length and batch size (keep small for performance)
        int64_t seq_len = 2 + (seq_len_byte % 14);    // 2-15
        int64_t batch_size = 1 + (batch_size_byte % 4); // 1-4

        // Create TransformerEncoderLayer options
        // Note: batch_first is not available in C++ API, default is (seq_len, batch, d_model)
        auto options = torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
            .dim_feedforward(dim_feedforward)
            .dropout(dropout);
        
        if (activation_idx == 0) {
            options.activation(torch::kReLU);
        } else {
            options.activation(torch::kGELU);
        }

        torch::nn::TransformerEncoderLayer encoder_layer(options);

        // Create input tensor with shape (seq_len, batch_size, d_model)
        // C++ API uses (seq_len, batch, d_model) format by default
        torch::Tensor src = torch::randn({seq_len, batch_size, d_model});

        // Use remaining fuzzer data to perturb the tensor values
        if (offset + 4 <= Size) {
            float scale = static_cast<float>(Data[offset]) / 128.0f;
            src = src * scale;
            offset++;
        }

        // Create src_mask (optional) - shape (seq_len, seq_len)
        torch::Tensor src_mask;
        if ((use_mask % 3) == 0) {
            // Additive attention mask
            src_mask = torch::zeros({seq_len, seq_len});
            // Create causal-like mask pattern based on fuzzer data
            for (int64_t i = 0; i < seq_len; i++) {
                for (int64_t j = 0; j < seq_len; j++) {
                    if (offset < Size) {
                        bool mask_val = (Data[offset % Size] % 4) == 0;
                        if (mask_val && j > i) {
                            src_mask[i][j] = -1e9f;
                        }
                    }
                }
            }
            offset++;
        }

        // Create src_key_padding_mask (optional) - shape (batch_size, seq_len)
        torch::Tensor src_key_padding_mask;
        if ((use_key_padding_mask % 3) == 0) {
            src_key_padding_mask = torch::zeros({batch_size, seq_len}, torch::kBool);
            // Don't mask all positions - ensure at least one position is unmasked
            for (int64_t b = 0; b < batch_size; b++) {
                for (int64_t s = 0; s < seq_len - 1; s++) { // Leave last position unmasked
                    if (offset < Size) {
                        bool pad = (Data[offset % Size] % 4) == 0;
                        src_key_padding_mask[b][s] = pad;
                        offset++;
                    }
                }
            }
        }

        // Forward pass
        torch::Tensor output;
        try {
            if (src_mask.defined() && src_key_padding_mask.defined()) {
                output = encoder_layer->forward(src, src_mask, src_key_padding_mask);
            } else if (src_mask.defined()) {
                output = encoder_layer->forward(src, src_mask);
            } else if (src_key_padding_mask.defined()) {
                output = encoder_layer->forward(src, {}, src_key_padding_mask);
            } else {
                output = encoder_layer->forward(src);
            }

            // Verify output shape matches input shape
            if (output.sizes() != src.sizes()) {
                std::cerr << "Shape mismatch!" << std::endl;
            }
        } catch (const c10::Error& e) {
            // Expected errors from invalid configurations - silently ignore
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}