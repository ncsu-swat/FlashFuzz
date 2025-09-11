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
        
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor for the transformer encoder
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input tensor has at least 3 dimensions for transformer encoder
        // (batch_size, sequence_length, embedding_dim)
        if (input_tensor.dim() < 3) {
            // Expand dimensions if needed
            while (input_tensor.dim() < 3) {
                input_tensor = input_tensor.unsqueeze(0);
            }
        }
        
        // Extract parameters for transformer encoder from the remaining data
        uint8_t num_layers = 1;
        int64_t d_model = 8;
        int64_t nhead = 2;
        int64_t dim_feedforward = 16;
        double dropout = 0.0;
        
        if (offset + 5 <= Size) {
            num_layers = Data[offset++] % 3 + 1; // 1-3 layers
            
            // Extract d_model (must be divisible by nhead)
            uint8_t d_model_factor = Data[offset++] % 8 + 1;
            d_model = d_model_factor * 8; // 8, 16, 24, ..., 64
            
            // Extract nhead (must divide d_model evenly)
            uint8_t nhead_options[] = {1, 2, 4, 8};
            nhead = nhead_options[Data[offset++] % 4];
            if (nhead > d_model) nhead = d_model;
            
            // Extract dim_feedforward
            uint8_t dim_ff_factor = Data[offset++] % 8 + 1;
            dim_feedforward = dim_ff_factor * 16; // 16, 32, 48, ..., 128
            
            // Extract dropout
            dropout = static_cast<double>(Data[offset++]) / 255.0;
        }
        
        // Ensure the input tensor's last dimension matches d_model
        if (input_tensor.size(-1) != d_model) {
            input_tensor = input_tensor.to(torch::kFloat);
            
            // Reshape the last dimension to match d_model
            std::vector<int64_t> new_shape = input_tensor.sizes().vec();
            new_shape[new_shape.size() - 1] = d_model;
            input_tensor = input_tensor.reshape(new_shape);
        }
        
        // Create encoder layer
        torch::nn::TransformerEncoderLayerOptions encoder_layer_options = 
            torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
                .dim_feedforward(dim_feedforward)
                .dropout(dropout)
                .activation(torch::kReLU);
        
        auto encoder_layer = torch::nn::TransformerEncoderLayer(encoder_layer_options);
        
        // Create transformer encoder
        torch::nn::TransformerEncoderOptions encoder_options = 
            torch::nn::TransformerEncoderOptions(encoder_layer, num_layers);
        
        auto transformer_encoder = torch::nn::TransformerEncoder(encoder_options);
        
        // Create source mask (optional)
        torch::Tensor src_mask;
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Create a square mask of size (sequence_length, sequence_length)
            int64_t seq_len = input_tensor.size(1);
            src_mask = torch::zeros({seq_len, seq_len});
            
            // Create a triangular mask (for causal attention)
            if (offset < Size && Data[offset++] % 2 == 0) {
                src_mask = torch::triu(torch::ones({seq_len, seq_len}) * -1e9, 1);
            }
        }
        
        // Create source padding mask (optional)
        torch::Tensor src_padding_mask;
        if (offset < Size && Data[offset++] % 2 == 0) {
            int64_t batch_size = input_tensor.size(0);
            int64_t seq_len = input_tensor.size(1);
            
            // Create random padding mask
            src_padding_mask = torch::zeros({batch_size, seq_len}, torch::kBool);
            
            // Randomly set some positions to be masked
            if (offset + batch_size * seq_len <= Size) {
                for (int64_t i = 0; i < batch_size; i++) {
                    for (int64_t j = 0; j < seq_len; j++) {
                        if (offset < Size && Data[offset++] % 4 == 0) {
                            src_padding_mask.index_put_({i, j}, true);
                        }
                    }
                }
            }
        }
        
        // Apply the transformer encoder
        torch::Tensor output;
        if (src_mask.defined() && src_padding_mask.defined()) {
            output = transformer_encoder->forward(input_tensor, src_mask, src_padding_mask);
        } else if (src_mask.defined()) {
            output = transformer_encoder->forward(input_tensor, src_mask);
        } else {
            output = transformer_encoder->forward(input_tensor);
        }
        
        // Verify output shape matches input shape
        assert(output.sizes() == input_tensor.sizes());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
