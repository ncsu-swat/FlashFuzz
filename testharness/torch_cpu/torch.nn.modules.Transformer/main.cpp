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
        torch::Tensor tgt;
        
        // Try to create target tensor if we have enough data
        if (offset < Size - 5) {
            tgt = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data, use src as tgt
            tgt = src.clone();
        }
        
        // Parse configuration parameters from the remaining data
        int64_t d_model = 16;
        int64_t nhead = 2;
        int64_t num_encoder_layers = 2;
        int64_t num_decoder_layers = 2;
        int64_t dim_feedforward = 64;
        double dropout = 0.0;
        
        // If we have more data, use it to configure the transformer
        if (offset + 6 < Size) {
            d_model = 8 + (Data[offset++] % 56); // 8-64
            nhead = 1 + (Data[offset++] % 8);    // 1-8
            num_encoder_layers = 1 + (Data[offset++] % 3); // 1-3
            num_decoder_layers = 1 + (Data[offset++] % 3); // 1-3
            dim_feedforward = d_model * (1 + (Data[offset++] % 4)); // d_model to 4*d_model
            dropout = static_cast<double>(Data[offset++]) / 255.0; // 0.0-1.0
        }
        
        // Ensure d_model is divisible by nhead
        d_model = (d_model / nhead) * nhead;
        if (d_model < nhead) d_model = nhead;
        
        // Create transformer module
        auto options = torch::nn::TransformerOptions(d_model, nhead)
            .num_encoder_layers(num_encoder_layers)
            .num_decoder_layers(num_decoder_layers)
            .dim_feedforward(dim_feedforward)
            .dropout(dropout);
        
        auto transformer = torch::nn::Transformer(options);
        
        // Reshape tensors if needed to match transformer requirements
        // Transformer expects [seq_len, batch, d_model] for both src and tgt
        
        // Reshape src tensor
        if (src.dim() == 0) {
            src = src.reshape({1, 1, 1}).expand({1, 1, d_model});
        } else if (src.dim() == 1) {
            int64_t seq_len = src.size(0);
            src = src.reshape({seq_len, 1, 1}).expand({seq_len, 1, d_model});
        } else if (src.dim() == 2) {
            int64_t seq_len = src.size(0);
            int64_t batch = src.size(1);
            src = src.reshape({seq_len, batch, 1}).expand({seq_len, batch, d_model});
        } else if (src.dim() >= 3) {
            int64_t seq_len = src.size(0);
            int64_t batch = src.size(1);
            // Keep first two dimensions, reshape remaining to d_model
            src = src.reshape({seq_len, batch, -1});
            if (src.size(2) != d_model) {
                src = src.expand({seq_len, batch, d_model});
            }
        }
        
        // Reshape tgt tensor
        if (tgt.dim() == 0) {
            tgt = tgt.reshape({1, 1, 1}).expand({1, 1, d_model});
        } else if (tgt.dim() == 1) {
            int64_t seq_len = tgt.size(0);
            tgt = tgt.reshape({seq_len, 1, 1}).expand({seq_len, 1, d_model});
        } else if (tgt.dim() == 2) {
            int64_t seq_len = tgt.size(0);
            int64_t batch = tgt.size(1);
            tgt = tgt.reshape({seq_len, batch, 1}).expand({seq_len, batch, d_model});
        } else if (tgt.dim() >= 3) {
            int64_t seq_len = tgt.size(0);
            int64_t batch = tgt.size(1);
            // Keep first two dimensions, reshape remaining to d_model
            tgt = tgt.reshape({seq_len, batch, -1});
            if (tgt.size(2) != d_model) {
                tgt = tgt.expand({seq_len, batch, d_model});
            }
        }
        
        // Convert tensors to float for transformer
        src = src.to(torch::kFloat);
        tgt = tgt.to(torch::kFloat);
        
        // Generate square subsequent mask for target
        auto tgt_mask = transformer->generate_square_subsequent_mask(tgt.size(0));
        
        // Apply transformer
        auto output = transformer->forward(src, tgt, tgt_mask);
        
        // Verify output shape
        if (output.dim() != 3 || 
            output.size(0) != tgt.size(0) || 
            output.size(1) != tgt.size(1) || 
            output.size(2) != d_model) {
            throw std::runtime_error("Transformer output has unexpected shape");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}