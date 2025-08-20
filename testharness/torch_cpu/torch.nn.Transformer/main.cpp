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
        
        // Create input tensors for the Transformer
        torch::Tensor src = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor tgt = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for the Transformer
        uint8_t d_model_byte = offset < Size ? Data[offset++] : 4;
        uint8_t nhead_byte = offset < Size ? Data[offset++] : 2;
        uint8_t num_encoder_layers_byte = offset < Size ? Data[offset++] : 1;
        uint8_t num_decoder_layers_byte = offset < Size ? Data[offset++] : 1;
        uint8_t dim_feedforward_byte = offset < Size ? Data[offset++] : 8;
        uint8_t dropout_byte = offset < Size ? Data[offset++] : 0;
        
        // Ensure parameters are within reasonable ranges
        int64_t d_model = 2 + (d_model_byte % 30);
        int64_t nhead = 1 + (nhead_byte % 8);
        int64_t num_encoder_layers = 1 + (num_encoder_layers_byte % 3);
        int64_t num_decoder_layers = 1 + (num_decoder_layers_byte % 3);
        int64_t dim_feedforward = d_model + (dim_feedforward_byte % 64);
        double dropout = static_cast<double>(dropout_byte) / 255.0;
        
        // Ensure d_model is divisible by nhead
        d_model = nhead * (d_model / nhead + (d_model % nhead > 0 ? 1 : 0));
        
        // Create the Transformer model
        torch::nn::TransformerOptions options(d_model, nhead);
        options.num_encoder_layers(num_encoder_layers);
        options.num_decoder_layers(num_decoder_layers);
        options.dim_feedforward(dim_feedforward);
        options.dropout(dropout);
        
        auto transformer = torch::nn::Transformer(options);
        
        // Reshape tensors if needed to match transformer requirements
        // Transformer expects [sequence_length, batch_size, d_model]
        if (src.dim() < 3) {
            int64_t seq_len = 2;
            int64_t batch_size = 1;
            src = src.reshape({seq_len, batch_size, d_model});
        } else {
            // Ensure the last dimension is d_model
            std::vector<int64_t> new_shape = src.sizes().vec();
            new_shape[new_shape.size() - 1] = d_model;
            src = src.reshape(new_shape);
        }
        
        if (tgt.dim() < 3) {
            int64_t seq_len = 2;
            int64_t batch_size = 1;
            tgt = tgt.reshape({seq_len, batch_size, d_model});
        } else {
            // Ensure the last dimension is d_model
            std::vector<int64_t> new_shape = tgt.sizes().vec();
            new_shape[new_shape.size() - 1] = d_model;
            tgt = tgt.reshape(new_shape);
        }
        
        // Create src_mask and tgt_mask (optional)
        torch::Tensor src_mask;
        torch::Tensor tgt_mask;
        
        if (offset + 1 < Size && Data[offset++] % 2 == 0) {
            // Create src_mask
            int64_t src_seq_len = src.size(0);
            src_mask = torch::zeros({src_seq_len, src_seq_len});
        }
        
        if (offset + 1 < Size && Data[offset++] % 2 == 0) {
            // Create tgt_mask (usually lower triangular for causal attention)
            int64_t tgt_seq_len = tgt.size(0);
            tgt_mask = torch::ones({tgt_seq_len, tgt_seq_len}).tril();
            tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, -1e9);
        }
        
        // Create memory_mask (optional)
        torch::Tensor memory_mask;
        if (offset + 1 < Size && Data[offset++] % 2 == 0) {
            int64_t tgt_seq_len = tgt.size(0);
            int64_t src_seq_len = src.size(0);
            memory_mask = torch::zeros({tgt_seq_len, src_seq_len});
        }
        
        // Create src_key_padding_mask and tgt_key_padding_mask (optional)
        torch::Tensor src_key_padding_mask;
        torch::Tensor tgt_key_padding_mask;
        torch::Tensor memory_key_padding_mask;
        
        if (offset + 1 < Size && Data[offset++] % 2 == 0) {
            int64_t batch_size = src.size(1);
            int64_t src_seq_len = src.size(0);
            src_key_padding_mask = torch::zeros({batch_size, src_seq_len}, torch::kBool);
        }
        
        if (offset + 1 < Size && Data[offset++] % 2 == 0) {
            int64_t batch_size = tgt.size(1);
            int64_t tgt_seq_len = tgt.size(0);
            tgt_key_padding_mask = torch::zeros({batch_size, tgt_seq_len}, torch::kBool);
        }
        
        if (offset + 1 < Size && Data[offset++] % 2 == 0) {
            int64_t batch_size = tgt.size(1);
            int64_t src_seq_len = src.size(0);
            memory_key_padding_mask = torch::zeros({batch_size, src_seq_len}, torch::kBool);
        }
        
        // Apply the transformer
        torch::Tensor output = transformer->forward(
            src, tgt, src_mask, tgt_mask, memory_mask,
            src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask
        );
        
        // Verify output shape
        auto output_sizes = output.sizes();
        auto tgt_sizes = tgt.sizes();
        
        // Basic sanity check on output
        if (output.numel() > 0) {
            // Access some values to ensure computation completed
            auto sum = output.sum().item<float>();
            (void)sum; // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}