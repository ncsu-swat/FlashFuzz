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
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor src = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get configuration parameters from the remaining data
        uint16_t d_model = 512;
        uint16_t nhead = 8;
        uint16_t dim_feedforward = 2048;
        float dropout = 0.1;
        bool batch_first = false;
        
        if (offset + 8 <= Size) {
            // Extract parameters from the input data
            std::memcpy(&d_model, Data + offset, sizeof(d_model));
            offset += sizeof(d_model);
            
            std::memcpy(&nhead, Data + offset, sizeof(nhead));
            offset += sizeof(nhead);
            
            std::memcpy(&dim_feedforward, Data + offset, sizeof(dim_feedforward));
            offset += sizeof(dim_feedforward);
            
            // Ensure d_model is divisible by nhead
            if (nhead == 0) nhead = 1;
            d_model = std::max<uint16_t>(1, d_model);
            if (d_model % nhead != 0) {
                d_model = nhead * (d_model / nhead + 1);
            }
            
            // Ensure reasonable values
            dim_feedforward = std::max<uint16_t>(1, dim_feedforward);
            
            // Get dropout rate (0.0 to 1.0)
            if (offset < Size) {
                dropout = static_cast<float>(Data[offset]) / 255.0f;
                offset++;
            }
            
            // Get batch_first flag
            if (offset < Size) {
                batch_first = (Data[offset] % 2) == 1;
                offset++;
            }
        }
        
        // Create TransformerEncoderLayer
        torch::nn::TransformerEncoderLayer encoder_layer(
            torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
                .dim_feedforward(dim_feedforward)
                .dropout(dropout)
                .activation(torch::kReLU)
                .batch_first(batch_first)
        );
        
        // Reshape input tensor if needed to match expected dimensions
        if (src.dim() < 2) {
            // For 0D or 1D tensors, reshape to 2D
            if (src.dim() == 0) {
                src = src.reshape({1, 1});
            } else {
                src = src.reshape({1, src.size(0)});
            }
        }
        
        // Ensure the embedding dimension matches d_model
        auto sizes = src.sizes().vec();
        int64_t seq_len_dim = batch_first ? 1 : 0;
        int64_t batch_dim = batch_first ? 0 : 1;
        int64_t embed_dim = 2;
        
        if (src.dim() == 2) {
            // Add embedding dimension
            src = src.unsqueeze(-1);
            sizes.push_back(1);
        }
        
        // Resize to match required dimensions
        if (sizes.size() > embed_dim && sizes[embed_dim] != d_model) {
            sizes[embed_dim] = d_model;
            src = src.resize_(sizes);
        }
        
        // Create src_mask (optional)
        torch::Tensor src_mask;
        if (offset < Size && Data[offset] % 2 == 0) {
            // Create a random mask
            int64_t seq_len = src.size(seq_len_dim);
            src_mask = torch::zeros({seq_len, seq_len});
            
            // Fill with some pattern based on remaining data
            for (int64_t i = 0; i < seq_len && offset + i < Size; i++) {
                for (int64_t j = 0; j < seq_len && offset + i*seq_len + j < Size; j++) {
                    if ((i <= j) && (offset + i*seq_len + j < Size)) {
                        src_mask[i][j] = (Data[offset + i*seq_len + j] % 2) ? 0.0 : -1e9;
                    }
                }
            }
        }
        
        // Create src_key_padding_mask (optional)
        torch::Tensor src_key_padding_mask;
        if (offset < Size && Data[offset] % 3 == 0) {
            int64_t batch_size = src.size(batch_dim);
            int64_t seq_len = src.size(seq_len_dim);
            src_key_padding_mask = torch::zeros({batch_size, seq_len}, torch::kBool);
            
            // Fill with some pattern based on remaining data
            for (int64_t i = 0; i < batch_size && offset + i < Size; i++) {
                for (int64_t j = 0; j < seq_len && offset + i*seq_len + j < Size; j++) {
                    if (offset + i*seq_len + j < Size) {
                        src_key_padding_mask[i][j] = (Data[offset + i*seq_len + j] % 2) == 1;
                    }
                }
            }
        }
        
        // Forward pass through the encoder layer
        torch::Tensor output;
        if (src_mask.defined() && src_key_padding_mask.defined()) {
            output = encoder_layer->forward(src, src_mask, src_key_padding_mask);
        } else if (src_mask.defined()) {
            output = encoder_layer->forward(src, src_mask);
        } else {
            output = encoder_layer->forward(src);
        }
        
        // Verify output shape matches input shape
        assert(output.sizes() == src.sizes());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
