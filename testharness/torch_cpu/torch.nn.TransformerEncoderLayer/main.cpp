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
        
        // Create input tensor
        torch::Tensor src = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for TransformerEncoderLayer from the remaining data
        int64_t d_model = 8;
        int64_t nhead = 2;
        int64_t dim_feedforward = 16;
        double dropout = 0.0;
        
        if (offset + 8 <= Size) {
            int64_t raw_d_model;
            std::memcpy(&raw_d_model, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            d_model = std::abs(raw_d_model) % 64 + 2;
            d_model = d_model - (d_model % 2); // Make sure d_model is divisible by nhead
        }
        
        if (offset + 8 <= Size) {
            int64_t raw_nhead;
            std::memcpy(&raw_nhead, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            nhead = std::abs(raw_nhead) % 8 + 1;
            
            // Ensure d_model is divisible by nhead
            d_model = (d_model / nhead) * nhead;
            if (d_model == 0) d_model = nhead;
        }
        
        if (offset + 8 <= Size) {
            int64_t raw_dim_feedforward;
            std::memcpy(&raw_dim_feedforward, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dim_feedforward = std::abs(raw_dim_feedforward) % 128 + 1;
        }
        
        if (offset + 8 <= Size) {
            double raw_dropout;
            std::memcpy(&raw_dropout, Data + offset, sizeof(double));
            offset += sizeof(double);
            dropout = std::abs(raw_dropout) / (1.0 + std::abs(raw_dropout)); // Normalize to [0, 1)
        }
        
        // Create activation function
        torch::nn::TransformerEncoderLayerOptions::activation_t activation = torch::kReLU;
        if (offset < Size) {
            uint8_t act_selector = Data[offset++];
            switch (act_selector % 2) {
                case 0: activation = torch::kReLU; break;
                case 1: activation = torch::kGELU; break;
            }
        }
        
        // Reshape src tensor if needed to match expected input shape for transformer
        if (src.dim() < 2) {
            // For 0D or 1D tensors, reshape to 2D
            src = src.reshape({1, src.numel()});
        }
        
        // Ensure the last dimension matches d_model
        auto src_sizes = src.sizes().vec();
        if (src_sizes.back() != d_model) {
            src_sizes.back() = d_model;
            src = src.reshape(src_sizes);
        }
        
        // Create the TransformerEncoderLayer
        torch::nn::TransformerEncoderLayerOptions options = 
            torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
                .dim_feedforward(dim_feedforward)
                .dropout(dropout)
                .activation(activation);
        
        auto encoder_layer = torch::nn::TransformerEncoderLayer(options);
        
        // Set to eval mode to avoid dropout randomness
        encoder_layer->eval();
        
        // Apply the encoder layer
        auto output = encoder_layer->forward(src);
        
        // Optional: Create a mask for the encoder
        if (offset < Size && Data[offset++] % 2 == 1) {
            torch::Tensor mask;
            
            // Determine sequence length
            int64_t seq_len = src.size(0);
            
            // Create a random mask
            if (offset < Size && Data[offset++] % 2 == 0) {
                // Create a boolean mask
                mask = torch::zeros({seq_len, seq_len}, torch::kBool);
                
                // Fill mask with some pattern based on remaining data
                for (int64_t i = 0; i < seq_len && offset < Size; i++) {
                    for (int64_t j = 0; j < seq_len && offset < Size; j++) {
                        if (offset < Size) {
                            mask[i][j] = Data[offset++] % 2 == 1;
                        }
                    }
                }
            } else {
                // Create an additive mask
                mask = torch::zeros({seq_len, seq_len}, torch::kFloat);
                
                // Fill mask with some pattern
                for (int64_t i = 0; i < seq_len && offset < Size; i++) {
                    for (int64_t j = 0; j < seq_len && offset < Size; j++) {
                        if (offset < Size) {
                            mask[i][j] = static_cast<float>(Data[offset++] % 100) * -0.1f;
                        }
                    }
                }
            }
            
            // Apply the encoder layer with mask
            output = encoder_layer->forward(src, mask);
        }
        
        // Try with src_key_padding_mask if we have enough data
        if (offset < Size && Data[offset++] % 2 == 1) {
            int64_t batch_size = src.size(1);
            int64_t seq_len = src.size(0);
            
            auto padding_mask = torch::zeros({batch_size, seq_len}, torch::kBool);
            
            // Fill padding mask with some pattern
            for (int64_t i = 0; i < batch_size && offset < Size; i++) {
                for (int64_t j = 0; j < seq_len && offset < Size; j++) {
                    if (offset < Size) {
                        padding_mask[i][j] = Data[offset++] % 2 == 1;
                    }
                }
            }
            
            // Apply the encoder layer with padding mask
            output = encoder_layer->forward(src, torch::Tensor(), padding_mask);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}