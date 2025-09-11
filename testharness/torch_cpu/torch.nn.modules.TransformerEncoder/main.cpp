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
        
        // Create input tensor
        torch::Tensor src;
        try {
            src = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception&) {
            return 0;
        }
        
        // Ensure src has at least 3 dimensions for transformer encoder
        // (batch_size, sequence_length, embedding_dim)
        if (src.dim() < 3) {
            if (src.dim() == 2) {
                // Add batch dimension
                src = src.unsqueeze(0);
            } else if (src.dim() == 1) {
                // Add batch and sequence dimensions
                src = src.unsqueeze(0).unsqueeze(0);
            } else {
                // Add batch, sequence, and embedding dimensions
                src = src.unsqueeze(0).unsqueeze(0).unsqueeze(0);
            }
        }
        
        // Extract parameters from the remaining data
        if (offset + 8 > Size) {
            return 0;
        }
        
        // Parse d_model (embedding dimension)
        int64_t d_model = 0;
        if (src.dim() >= 3) {
            d_model = src.size(2);
        } else {
            d_model = 64; // Default value
        }
        
        // Parse nhead (number of attention heads)
        uint8_t nhead_byte = Data[offset++];
        int64_t nhead = (nhead_byte % 8) + 1; // 1-8 heads
        
        // Parse num_encoder_layers
        uint8_t num_layers_byte = Data[offset++];
        int64_t num_encoder_layers = (num_layers_byte % 3) + 1; // 1-3 layers
        
        // Parse dim_feedforward
        int64_t dim_feedforward = 0;
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&dim_feedforward, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dim_feedforward = std::abs(dim_feedforward) % 1024 + 64; // Reasonable range
        } else {
            dim_feedforward = d_model * 4; // Default value
        }
        
        // Parse dropout
        double dropout = 0.0;
        if (offset < Size) {
            dropout = static_cast<double>(Data[offset++]) / 255.0; // 0.0-1.0
        }
        
        // Parse activation function
        torch::nn::functional::ActivationFuncOptions::activation_t activation = torch::kReLU;
        if (offset < Size) {
            uint8_t activation_byte = Data[offset++];
            if (activation_byte % 2 == 1) {
                activation = torch::kGELU;
            } else {
                activation = torch::kReLU;
            }
        }
        
        // Parse layer_norm_eps
        double layer_norm_eps = 1e-5;
        if (offset < Size) {
            uint8_t eps_byte = Data[offset++];
            layer_norm_eps = 1e-5 * (1 + (eps_byte % 10)); // Range of reasonable epsilon values
        }
        
        // Parse batch_first
        bool batch_first = false;
        if (offset < Size) {
            batch_first = (Data[offset++] % 2 == 1);
        }
        
        // Parse norm_first
        bool norm_first = false;
        if (offset < Size) {
            norm_first = (Data[offset++] % 2 == 1);
        }
        
        // Create encoder layer
        auto encoder_layer = torch::nn::TransformerEncoderLayer(
            torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
                .dim_feedforward(dim_feedforward)
                .dropout(dropout)
                .activation(activation)
                .layer_norm_eps(layer_norm_eps)
                .batch_first(batch_first)
                .norm_first(norm_first));
        
        // Create transformer encoder
        auto transformer_encoder = torch::nn::TransformerEncoder(
            torch::nn::TransformerEncoderOptions(encoder_layer, num_encoder_layers));
        
        // Create src_mask (optional)
        torch::Tensor src_mask;
        bool use_mask = false;
        if (offset < Size) {
            use_mask = (Data[offset++] % 2 == 1);
            if (use_mask) {
                try {
                    src_mask = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Ensure mask has correct shape if provided
                    int64_t seq_len = src.size(batch_first ? 1 : 0);
                    if (src_mask.dim() != 2 || src_mask.size(0) != seq_len || src_mask.size(1) != seq_len) {
                        // Create a valid mask
                        src_mask = torch::zeros({seq_len, seq_len});
                    }
                } catch (const std::exception&) {
                    use_mask = false;
                }
            }
        }
        
        // Create src_key_padding_mask (optional)
        torch::Tensor src_key_padding_mask;
        bool use_key_padding_mask = false;
        if (offset < Size) {
            use_key_padding_mask = (Data[offset++] % 2 == 1);
            if (use_key_padding_mask) {
                try {
                    src_key_padding_mask = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Ensure key padding mask has correct shape if provided
                    int64_t batch_size = src.size(batch_first ? 0 : 1);
                    int64_t seq_len = src.size(batch_first ? 1 : 0);
                    if (src_key_padding_mask.dim() != 2 || 
                        src_key_padding_mask.size(0) != batch_size || 
                        src_key_padding_mask.size(1) != seq_len) {
                        // Create a valid key padding mask
                        src_key_padding_mask = torch::zeros({batch_size, seq_len});
                    }
                } catch (const std::exception&) {
                    use_key_padding_mask = false;
                }
            }
        }
        
        // Apply transformer encoder
        torch::Tensor output;
        if (use_mask && use_key_padding_mask) {
            output = transformer_encoder->forward(src, src_mask, src_key_padding_mask);
        } else if (use_mask) {
            output = transformer_encoder->forward(src, src_mask);
        } else if (use_key_padding_mask) {
            output = transformer_encoder->forward(src, torch::Tensor(), src_key_padding_mask);
        } else {
            output = transformer_encoder->forward(src);
        }
        
        // Check output
        if (output.numel() > 0) {
            auto sum = output.sum().item<float>();
            if (std::isnan(sum) || std::isinf(sum)) {
                return 0;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
