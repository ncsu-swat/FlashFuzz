#include "fuzzer_utils.h"
#include <iostream>

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
        if (Size < 20) {
            return 0;
        }

        size_t offset = 0;

        // Extract parameters from fuzz data
        int64_t d_model = 8;
        int64_t nhead = 2;
        int64_t dim_feedforward = 16;
        int64_t seq_len = 4;
        int64_t batch_size = 2;

        if (offset + 1 <= Size) {
            nhead = (Data[offset++] % 4) + 1;  // 1-4 heads
        }

        if (offset + 1 <= Size) {
            // d_model must be divisible by nhead
            int64_t multiplier = (Data[offset++] % 8) + 1;  // 1-8
            d_model = nhead * multiplier;
        }

        if (offset + 1 <= Size) {
            dim_feedforward = (Data[offset++] % 32) + 4;  // 4-35
        }

        if (offset + 1 <= Size) {
            seq_len = (Data[offset++] % 8) + 1;  // 1-8
        }

        if (offset + 1 <= Size) {
            batch_size = (Data[offset++] % 4) + 1;  // 1-4
        }

        // Extract dropout (set to 0 for deterministic behavior in eval mode)
        double dropout = 0.0;

        // Select activation function as string
        std::string activation_str = "relu";
        if (offset < Size) {
            uint8_t act_selector = Data[offset++];
            switch (act_selector % 2) {
                case 0: activation_str = "relu"; break;
                case 1: activation_str = "gelu"; break;
            }
        }

        // Create input tensor with correct shape: (seq_len, batch_size, d_model)
        torch::Tensor src = torch::randn({seq_len, batch_size, d_model});

        // Fill tensor with fuzz data if available
        if (offset + sizeof(float) <= Size) {
            auto src_accessor = src.accessor<float, 3>();
            for (int64_t s = 0; s < seq_len && offset + sizeof(float) <= Size; s++) {
                for (int64_t b = 0; b < batch_size && offset + sizeof(float) <= Size; b++) {
                    for (int64_t e = 0; e < d_model && offset + sizeof(float) <= Size; e++) {
                        float val;
                        std::memcpy(&val, Data + offset, sizeof(float));
                        offset += sizeof(float);
                        // Clamp to reasonable range
                        if (std::isfinite(val)) {
                            src_accessor[s][b][e] = std::clamp(val, -10.0f, 10.0f);
                        }
                    }
                }
            }
        }

        // Create TransformerEncoderLayer options
        torch::nn::TransformerEncoderLayerOptions options(d_model, nhead);
        options.dim_feedforward(dim_feedforward);
        options.dropout(dropout);
        options.activation(activation_str);

        // Optionally set batch_first
        bool batch_first = false;
        if (offset < Size) {
            batch_first = Data[offset++] % 2 == 1;
            options.batch_first(batch_first);
            
            // Adjust tensor shape if batch_first
            if (batch_first) {
                src = src.transpose(0, 1);  // Now shape: (batch_size, seq_len, d_model)
            }
        }

        // Create the TransformerEncoderLayer
        auto encoder_layer = torch::nn::TransformerEncoderLayer(options);
        encoder_layer->eval();  // Set to eval mode for deterministic behavior

        // Basic forward pass
        torch::Tensor output;
        try {
            output = encoder_layer->forward(src);
        } catch (const c10::Error&) {
            // Expected for invalid configurations
            return 0;
        }

        // Determine dimensions based on batch_first setting
        int64_t actual_seq_len = batch_first ? src.size(1) : src.size(0);
        int64_t actual_batch_size = batch_first ? src.size(0) : src.size(1);

        // Try with attention mask
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                // Create square attention mask: (seq_len, seq_len)
                torch::Tensor src_mask = torch::zeros({actual_seq_len, actual_seq_len});
                
                // Fill with additive mask values (negative = masked)
                for (int64_t i = 0; i < actual_seq_len && offset < Size; i++) {
                    for (int64_t j = 0; j < actual_seq_len && offset < Size; j++) {
                        if (Data[offset++] % 4 == 0) {
                            src_mask[i][j] = -std::numeric_limits<float>::infinity();
                        }
                    }
                }

                output = encoder_layer->forward(src, src_mask);
            } catch (const c10::Error&) {
                // Expected for some mask configurations
            }
        }

        // Try with key padding mask
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                // Create key padding mask: (batch_size, seq_len)
                torch::Tensor src_key_padding_mask = torch::zeros(
                    {actual_batch_size, actual_seq_len}, torch::kBool);
                
                // Fill with padding values
                for (int64_t b = 0; b < actual_batch_size && offset < Size; b++) {
                    for (int64_t s = 0; s < actual_seq_len && offset < Size; s++) {
                        src_key_padding_mask[b][s] = Data[offset++] % 4 == 0;
                    }
                }

                output = encoder_layer->forward(src, {}, src_key_padding_mask);
            } catch (const c10::Error&) {
                // Expected for some mask configurations
            }
        }

        // Try with both masks
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                torch::Tensor src_mask = torch::zeros({actual_seq_len, actual_seq_len});
                torch::Tensor src_key_padding_mask = torch::zeros(
                    {actual_batch_size, actual_seq_len}, torch::kBool);

                // Fill masks
                for (int64_t i = 0; i < actual_seq_len && offset < Size; i++) {
                    for (int64_t j = 0; j < actual_seq_len && offset < Size; j++) {
                        if (Data[offset++] % 4 == 0) {
                            src_mask[i][j] = -std::numeric_limits<float>::infinity();
                        }
                    }
                }
                for (int64_t b = 0; b < actual_batch_size && offset < Size; b++) {
                    for (int64_t s = 0; s < actual_seq_len && offset < Size; s++) {
                        src_key_padding_mask[b][s] = Data[offset++] % 4 == 0;
                    }
                }

                output = encoder_layer->forward(src, src_mask, src_key_padding_mask);
            } catch (const c10::Error&) {
                // Expected for some configurations
            }
        }

        // Test with different layer norm epsilon if available
        if (offset < Size && Data[offset++] % 5 == 0) {
            try {
                torch::nn::TransformerEncoderLayerOptions opts2(d_model, nhead);
                opts2.dim_feedforward(dim_feedforward);
                opts2.dropout(0.0);
                opts2.batch_first(batch_first);
                opts2.layer_norm_eps(1e-6);  // Different epsilon
                
                auto encoder_layer2 = torch::nn::TransformerEncoderLayer(opts2);
                encoder_layer2->eval();
                
                output = encoder_layer2->forward(src);
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