#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensors for transformer
        torch::Tensor src = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a second tensor for target if there's data left
        torch::Tensor tgt;
        if (offset + 5 < Size) {
            tgt = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data for second tensor, use the first one
            tgt = src.clone();
        }
        
        // Extract configuration parameters from remaining data
        int64_t d_model = 32;
        int64_t nhead = 8;
        int64_t num_encoder_layers = 2;
        int64_t num_decoder_layers = 2;
        int64_t dim_feedforward = 128;
        double dropout = 0.0;
        
        // If we have more data, use it to configure the transformer
        if (offset + 6 < Size) {
            d_model = 8 + (Data[offset++] % 56); // 8-64 range
            nhead = 1 + (Data[offset++] % 8);    // 1-8 range
            num_encoder_layers = 1 + (Data[offset++] % 3); // 1-4 range
            num_decoder_layers = 1 + (Data[offset++] % 3); // 1-4 range
            dim_feedforward = d_model * (1 + (Data[offset++] % 4)); // d_model to 5*d_model
            dropout = static_cast<double>(Data[offset++]) / 255.0; // 0.0-1.0 range
        }
        
        // Ensure d_model is divisible by nhead
        d_model = (d_model / nhead) * nhead;
        if (d_model == 0) d_model = nhead;
        
        // Create transformer model
        torch::nn::TransformerOptions transformer_options(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers
        );
        
        transformer_options.dim_feedforward(dim_feedforward);
        transformer_options.dropout(dropout);
        
        // Add activation function option if we have more data
        if (offset < Size) {
            uint8_t activation_selector = Data[offset++] % 2;
            if (activation_selector == 0) {
                transformer_options.activation(torch::kReLU);
            } else {
                transformer_options.activation(torch::kGELU);
            }
        }
        
        // Create the transformer module
        torch::nn::Transformer transformer_module(transformer_options);
        
        // Reshape tensors if needed to match transformer requirements
        // Transformer expects [sequence_length, batch_size, d_model]
        if (src.dim() == 0) {
            src = src.unsqueeze(0).unsqueeze(0).unsqueeze(0);
            if (src.size(2) != d_model) {
                src = src.expand({1, 1, d_model});
            }
        } else if (src.dim() == 1) {
            src = src.unsqueeze(0).unsqueeze(1);
            if (src.size(2) != d_model) {
                src = src.expand({src.size(0), 1, d_model});
            }
        } else if (src.dim() == 2) {
            src = src.unsqueeze(1);
            if (src.size(2) != d_model) {
                src = src.expand({src.size(0), src.size(1), d_model});
            }
        } else if (src.dim() >= 3) {
            // Keep first dimension as sequence length, second as batch size
            // and reshape the rest to match d_model
            auto old_shape = src.sizes().vec();
            std::vector<int64_t> new_shape = {old_shape[0], old_shape[1], d_model};
            src = src.reshape(new_shape);
        }
        
        // Similar reshaping for target tensor
        if (tgt.dim() == 0) {
            tgt = tgt.unsqueeze(0).unsqueeze(0).unsqueeze(0);
            if (tgt.size(2) != d_model) {
                tgt = tgt.expand({1, 1, d_model});
            }
        } else if (tgt.dim() == 1) {
            tgt = tgt.unsqueeze(0).unsqueeze(1);
            if (tgt.size(2) != d_model) {
                tgt = tgt.expand({tgt.size(0), 1, d_model});
            }
        } else if (tgt.dim() == 2) {
            tgt = tgt.unsqueeze(1);
            if (tgt.size(2) != d_model) {
                tgt = tgt.expand({tgt.size(0), tgt.size(1), d_model});
            }
        } else if (tgt.dim() >= 3) {
            auto old_shape = tgt.sizes().vec();
            std::vector<int64_t> new_shape = {old_shape[0], old_shape[1], d_model};
            tgt = tgt.reshape(new_shape);
        }
        
        // Convert tensors to float for transformer
        src = src.to(torch::kFloat);
        tgt = tgt.to(torch::kFloat);
        
        // Create masks if we have more data
        torch::Tensor src_mask, tgt_mask, memory_mask;
        torch::Tensor src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask;
        
        if (offset < Size) {
            // Create source mask (optional)
            if (Data[offset++] % 2 == 0) {
                src_mask = torch::zeros({src.size(0), src.size(0)});
            }
            
            // Create target mask (optional)
            if (offset < Size && Data[offset++] % 2 == 0) {
                tgt_mask = torch::zeros({tgt.size(0), tgt.size(0)});
                // Make it a causal mask (lower triangular)
                tgt_mask = torch::triu(torch::ones({tgt.size(0), tgt.size(0)}) * -1e9, 1);
            }
            
            // Create memory mask (optional)
            if (offset < Size && Data[offset++] % 2 == 0) {
                memory_mask = torch::zeros({tgt.size(0), src.size(0)});
            }
            
            // Create padding masks (optional)
            if (offset < Size && Data[offset++] % 2 == 0) {
                src_key_padding_mask = torch::zeros({src.size(1), src.size(0)}, torch::kBool);
            }
            
            if (offset < Size && Data[offset++] % 2 == 0) {
                tgt_key_padding_mask = torch::zeros({tgt.size(1), tgt.size(0)}, torch::kBool);
            }
            
            if (offset < Size && Data[offset++] % 2 == 0) {
                memory_key_padding_mask = torch::zeros({src.size(1), src.size(0)}, torch::kBool);
            }
        }
        
        // Forward pass through transformer
        torch::Tensor output = transformer_module->forward(
            src, tgt, src_mask, tgt_mask, memory_mask,
            src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask
        );
        
        // Test individual components if we have more data
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Test encoder
            torch::Tensor encoder_output = transformer_module.encoder->forward(
                src, src_mask, src_key_padding_mask
            );
            
            // Test decoder
            torch::Tensor decoder_output = transformer_module.decoder->forward(
                tgt, encoder_output, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask
            );
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
