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
        torch::NoGradGuard no_grad;
        
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 16) {
            return 0;
        }
        
        // Extract parameters for transformer encoder from the data
        uint8_t num_layers = (Data[offset++] % 3) + 1; // 1-3 layers
        
        // Extract d_model (must be divisible by nhead)
        uint8_t d_model_factor = (Data[offset++] % 4) + 1;
        int64_t d_model = d_model_factor * 8; // 8, 16, 24, 32
        
        // Extract nhead (must divide d_model evenly)
        int64_t nhead_options[] = {1, 2, 4, 8};
        int64_t nhead = nhead_options[Data[offset++] % 4];
        // Ensure nhead divides d_model
        while (d_model % nhead != 0) {
            nhead = nhead / 2;
            if (nhead < 1) nhead = 1;
        }
        
        // Extract dim_feedforward
        uint8_t dim_ff_factor = (Data[offset++] % 4) + 1;
        int64_t dim_feedforward = dim_ff_factor * 16; // 16, 32, 48, 64
        
        // Extract batch size and sequence length
        int64_t batch_size = (Data[offset++] % 4) + 1; // 1-4
        int64_t seq_len = (Data[offset++] % 8) + 1;    // 1-8
        
        // Extract flags for optional masks
        bool use_src_mask = Data[offset++] % 2 == 0;
        bool use_causal_mask = Data[offset++] % 2 == 0;
        bool use_padding_mask = Data[offset++] % 2 == 0;
        
        // Create input tensor with correct shape: (seq_len, batch_size, d_model) for batch_first=false (default)
        torch::Tensor input_tensor = torch::randn({seq_len, batch_size, d_model});
        
        // Use remaining fuzz data to perturb the tensor if available
        if (offset + 4 <= Size) {
            float scale = static_cast<float>(Data[offset++]) / 25.5f;
            input_tensor = input_tensor * scale;
        }
        
        // Create encoder layer options (batch_first is not available in C++ API, defaults to false)
        auto encoder_layer_options = 
            torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
                .dim_feedforward(dim_feedforward)
                .dropout(0.0);  // Disable dropout for deterministic fuzzing
        
        // Create encoder layer
        auto encoder_layer = torch::nn::TransformerEncoderLayer(encoder_layer_options);
        
        // Create transformer encoder options (enable_nested_tensor not available in C++ API)
        auto encoder_options = 
            torch::nn::TransformerEncoderOptions(encoder_layer, num_layers);
        
        // Create transformer encoder
        auto transformer_encoder = torch::nn::TransformerEncoder(encoder_options);
        transformer_encoder->eval();  // Set to eval mode
        
        // Create source mask (optional) - shape: (seq_len, seq_len)
        torch::Tensor src_mask;
        if (use_src_mask) {
            if (use_causal_mask) {
                // Create a causal (triangular) mask
                src_mask = torch::triu(
                    torch::full({seq_len, seq_len}, -std::numeric_limits<float>::infinity()),
                    1
                );
            } else {
                // Create a random attention mask
                src_mask = torch::zeros({seq_len, seq_len});
            }
        }
        
        // Create source key padding mask (optional) - shape: (batch_size, seq_len)
        torch::Tensor src_key_padding_mask;
        if (use_padding_mask) {
            src_key_padding_mask = torch::zeros({batch_size, seq_len}, torch::kBool);
            
            // Randomly mask some positions based on fuzz data, but never mask all
            for (int64_t i = 0; i < batch_size && offset < Size; i++) {
                int64_t mask_count = 0;
                for (int64_t j = 0; j < seq_len && offset < Size; j++) {
                    // Don't mask if this would mask everything in the sequence
                    if (Data[offset++] % 8 == 0 && mask_count < seq_len - 1) {
                        src_key_padding_mask.index_put_({i, j}, true);
                        mask_count++;
                    }
                }
            }
        }
        
        // Apply the transformer encoder with different argument combinations
        torch::Tensor output;
        try {
            if (src_mask.defined() && src_key_padding_mask.defined()) {
                output = transformer_encoder->forward(input_tensor, src_mask, src_key_padding_mask);
            } else if (src_mask.defined()) {
                output = transformer_encoder->forward(input_tensor, src_mask);
            } else if (src_key_padding_mask.defined()) {
                output = transformer_encoder->forward(input_tensor, {}, src_key_padding_mask);
            } else {
                output = transformer_encoder->forward(input_tensor);
            }
            
            // Verify output shape matches input shape (seq_len, batch_size, d_model)
            if (output.sizes() != input_tensor.sizes()) {
                std::cerr << "Shape mismatch!" << std::endl;
            }
        } catch (const c10::Error&) {
            // Expected failures from invalid configurations - silently catch
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}