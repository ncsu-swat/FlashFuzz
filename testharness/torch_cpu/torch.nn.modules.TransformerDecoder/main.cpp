#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        size_t offset = 0;
        
        // Early exit for very small inputs
        if (Size < 20) {
            return 0;
        }
        
        // Parse configuration parameters from the input data
        // nhead must divide d_model, so we pick nhead first then make d_model a multiple
        uint8_t nhead = (Data[offset++] % 4) + 1;  // 1-4 heads
        uint8_t d_model_mult = (Data[offset++] % 8) + 1;  // 1-8 multiplier
        int64_t d_model = nhead * d_model_mult;  // Ensures d_model % nhead == 0
        
        uint8_t num_decoder_layers = (Data[offset++] % 3) + 1;  // 1-3 layers
        int64_t dim_feedforward = (Data[offset++] % 32) + 16;  // 16-47
        float dropout_rate = static_cast<float>(Data[offset++]) / 512.0f;  // 0-0.5 range
        
        // Sequence lengths and batch size
        int64_t batch_size = (Data[offset++] % 4) + 1;  // 1-4
        int64_t tgt_seq_len = (Data[offset++] % 8) + 1;  // 1-8
        int64_t memory_seq_len = (Data[offset++] % 8) + 1;  // 1-8
        
        // Create decoder layer
        torch::nn::TransformerDecoderLayerOptions decoder_layer_options = 
            torch::nn::TransformerDecoderLayerOptions(d_model, nhead)
                .dim_feedforward(dim_feedforward)
                .dropout(dropout_rate);
        
        auto decoder_layer = torch::nn::TransformerDecoderLayer(decoder_layer_options);
        
        // Create transformer decoder
        torch::nn::TransformerDecoderOptions decoder_options = 
            torch::nn::TransformerDecoderOptions(decoder_layer, num_decoder_layers);
        
        auto transformer_decoder = torch::nn::TransformerDecoder(decoder_options);
        
        // Set to eval mode to disable dropout for deterministic behavior
        transformer_decoder->eval();
        
        // Create tgt tensor (target sequence) with shape (tgt_seq_len, batch_size, d_model)
        // Note: PyTorch transformer expects (seq_len, batch, d_model) by default
        auto tgt = torch::randn({tgt_seq_len, batch_size, d_model});
        
        // Use fuzz data to perturb the tensor
        if (offset + 4 <= Size) {
            float scale = static_cast<float>(Data[offset++]) / 128.0f;
            tgt = tgt * scale;
        }
        
        // Create memory tensor (encoder output) with shape (memory_seq_len, batch_size, d_model)
        auto memory = torch::randn({memory_seq_len, batch_size, d_model});
        
        // Use fuzz data to perturb the tensor
        if (offset + 4 <= Size) {
            float scale = static_cast<float>(Data[offset++]) / 128.0f;
            memory = memory * scale;
        }
        
        // Create optional mask tensors
        torch::Tensor tgt_mask;
        torch::Tensor memory_mask;
        torch::Tensor tgt_key_padding_mask;
        torch::Tensor memory_key_padding_mask;
        
        // Randomly decide whether to use masks based on remaining data
        if (offset < Size && (Data[offset++] % 3) == 0) {
            // Create tgt_mask (square matrix of size tgt_seq_len x tgt_seq_len)
            // Use causal mask (upper triangular)
            tgt_mask = torch::triu(
                torch::ones({tgt_seq_len, tgt_seq_len}) * (-std::numeric_limits<float>::infinity()),
                1
            );
        }
        
        if (offset < Size && (Data[offset++] % 3) == 0) {
            // Create memory_mask (matrix of size tgt_seq_len x memory_seq_len)
            memory_mask = torch::zeros({tgt_seq_len, memory_seq_len});
            // Randomly mask some positions
            if (offset < Size) {
                float mask_prob = static_cast<float>(Data[offset++]) / 255.0f;
                auto mask_vals = torch::rand({tgt_seq_len, memory_seq_len});
                memory_mask = torch::where(
                    mask_vals < mask_prob,
                    torch::full({tgt_seq_len, memory_seq_len}, -std::numeric_limits<float>::infinity()),
                    torch::zeros({tgt_seq_len, memory_seq_len})
                );
            }
        }
        
        if (offset < Size && (Data[offset++] % 3) == 0) {
            // Create tgt_key_padding_mask (matrix of size batch x tgt_seq_len)
            tgt_key_padding_mask = torch::zeros({batch_size, tgt_seq_len}, torch::kBool);
            // Randomly set some padding positions
            if (offset < Size) {
                int num_padded = Data[offset++] % tgt_seq_len;
                for (int i = 0; i < num_padded && i < tgt_seq_len; i++) {
                    tgt_key_padding_mask.index_put_({torch::indexing::Slice(), tgt_seq_len - 1 - i}, true);
                }
            }
        }
        
        if (offset < Size && (Data[offset++] % 3) == 0) {
            // Create memory_key_padding_mask (matrix of size batch x memory_seq_len)
            memory_key_padding_mask = torch::zeros({batch_size, memory_seq_len}, torch::kBool);
            // Randomly set some padding positions
            if (offset < Size) {
                int num_padded = Data[offset++] % memory_seq_len;
                for (int i = 0; i < num_padded && i < memory_seq_len; i++) {
                    memory_key_padding_mask.index_put_({torch::indexing::Slice(), memory_seq_len - 1 - i}, true);
                }
            }
        }
        
        // Apply the transformer decoder with inner try-catch for shape issues
        torch::Tensor output;
        try {
            output = transformer_decoder->forward(
                tgt, 
                memory, 
                tgt_mask.defined() ? tgt_mask : torch::Tensor(),
                memory_mask.defined() ? memory_mask : torch::Tensor(),
                tgt_key_padding_mask.defined() ? tgt_key_padding_mask : torch::Tensor(),
                memory_key_padding_mask.defined() ? memory_key_padding_mask : torch::Tensor()
            );
        } catch (const c10::Error&) {
            // Shape mismatches or other tensor errors - silently discard
            return 0;
        }
        
        // Verify output shape matches input shape
        if (output.sizes() != tgt.sizes()) {
            // Unexpected shape mismatch - this would be a real bug
            std::cerr << "Output shape mismatch: expected " << tgt.sizes() 
                      << " got " << output.sizes() << std::endl;
        }
        
        // Additional operations to increase coverage
        if (offset < Size && (Data[offset++] % 2) == 0) {
            // Test backward pass
            try {
                auto loss = output.sum();
                loss.backward();
            } catch (const c10::Error&) {
                // Gradient computation may fail - silently handle
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}