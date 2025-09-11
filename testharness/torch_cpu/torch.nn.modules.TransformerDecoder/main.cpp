#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Early exit for very small inputs
        if (Size < 10) {
            return 0;
        }
        
        // Parse configuration parameters from the input data
        uint8_t d_model = Data[offset++] % 32 + 1;
        uint8_t nhead = Data[offset++] % 8 + 1;
        uint8_t num_decoder_layers = Data[offset++] % 4 + 1;
        uint8_t dim_feedforward = Data[offset++] % 64 + 16;
        float dropout_rate = static_cast<float>(Data[offset++]) / 255.0f;
        
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
        
        // Create tgt tensor (target sequence)
        auto tgt = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tgt has at least 3 dimensions (batch, seq_len, d_model)
        if (tgt.dim() < 3) {
            tgt = tgt.reshape({1, 1, tgt.numel()});
        }
        
        // Ensure the last dimension is d_model
        auto tgt_shape = tgt.sizes().vec();
        if (tgt_shape.back() != d_model) {
            tgt_shape.back() = d_model;
            tgt = tgt.reshape(tgt_shape);
        }
        
        // Create memory tensor (encoder output)
        auto memory = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure memory has at least 3 dimensions (batch, seq_len, d_model)
        if (memory.dim() < 3) {
            memory = memory.reshape({1, 1, memory.numel()});
        }
        
        // Ensure the last dimension is d_model
        auto memory_shape = memory.sizes().vec();
        if (memory_shape.back() != d_model) {
            memory_shape.back() = d_model;
            memory = memory.reshape(memory_shape);
        }
        
        // Create optional mask tensors
        torch::Tensor tgt_mask;
        torch::Tensor memory_mask;
        torch::Tensor tgt_key_padding_mask;
        torch::Tensor memory_key_padding_mask;
        
        // Randomly decide whether to use masks based on remaining data
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Create tgt_mask (square matrix of size tgt_seq_len x tgt_seq_len)
            int64_t tgt_seq_len = tgt.size(1);
            tgt_mask = fuzzer_utils::createTensor(Data, Size, offset);
            tgt_mask = tgt_mask.reshape({tgt_seq_len, tgt_seq_len});
        }
        
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Create memory_mask (matrix of size tgt_seq_len x memory_seq_len)
            int64_t tgt_seq_len = tgt.size(1);
            int64_t memory_seq_len = memory.size(1);
            memory_mask = fuzzer_utils::createTensor(Data, Size, offset);
            memory_mask = memory_mask.reshape({tgt_seq_len, memory_seq_len});
        }
        
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Create tgt_key_padding_mask (matrix of size batch x tgt_seq_len)
            int64_t batch_size = tgt.size(0);
            int64_t tgt_seq_len = tgt.size(1);
            tgt_key_padding_mask = fuzzer_utils::createTensor(Data, Size, offset);
            tgt_key_padding_mask = tgt_key_padding_mask.reshape({batch_size, tgt_seq_len});
            tgt_key_padding_mask = tgt_key_padding_mask.to(torch::kBool);
        }
        
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Create memory_key_padding_mask (matrix of size batch x memory_seq_len)
            int64_t batch_size = memory.size(0);
            int64_t memory_seq_len = memory.size(1);
            memory_key_padding_mask = fuzzer_utils::createTensor(Data, Size, offset);
            memory_key_padding_mask = memory_key_padding_mask.reshape({batch_size, memory_seq_len});
            memory_key_padding_mask = memory_key_padding_mask.to(torch::kBool);
        }
        
        // Apply the transformer decoder
        torch::Tensor output = transformer_decoder->forward(
            tgt, 
            memory, 
            tgt_mask, 
            memory_mask, 
            tgt_key_padding_mask, 
            memory_key_padding_mask
        );
        
        // Verify output shape matches input shape
        if (output.sizes() != tgt.sizes()) {
            throw std::runtime_error("Output shape doesn't match input shape");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
