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
        
        // Parse input parameters for TransformerDecoderLayer
        int64_t d_model = (Data[0] % 32) + 1;
        int64_t nhead = (Data[1] % 8) + 1;
        int64_t dim_feedforward = ((Data[2] << 8) | Data[3]) % 1024 + 1;
        double dropout = static_cast<double>(Data[4]) / 255.0;
        bool batch_first = Data[6] & 1;
        bool norm_first = Data[7] & 1;
        
        offset = 8;
        
        // Create TransformerDecoderLayer
        torch::nn::TransformerDecoderLayer decoder_layer(
            torch::nn::TransformerDecoderLayerOptions(d_model, nhead)
                .dim_feedforward(dim_feedforward)
                .dropout(dropout)
                .activation(torch::kReLU)
                .batch_first(batch_first)
                .norm_first(norm_first)
        );
        
        // Create tgt tensor
        torch::Tensor tgt;
        if (offset < Size) {
            tgt = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create memory tensor
        torch::Tensor memory;
        if (offset < Size) {
            memory = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create optional tgt_mask tensor
        torch::Tensor tgt_mask;
        bool use_tgt_mask = offset < Size && (Data[offset++] & 1);
        if (use_tgt_mask && offset < Size) {
            tgt_mask = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Create optional memory_mask tensor
        torch::Tensor memory_mask;
        bool use_memory_mask = offset < Size && (Data[offset++] & 1);
        if (use_memory_mask && offset < Size) {
            memory_mask = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Create optional tgt_key_padding_mask tensor
        torch::Tensor tgt_key_padding_mask;
        bool use_tgt_key_padding_mask = offset < Size && (Data[offset++] & 1);
        if (use_tgt_key_padding_mask && offset < Size) {
            tgt_key_padding_mask = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Create optional memory_key_padding_mask tensor
        torch::Tensor memory_key_padding_mask;
        bool use_memory_key_padding_mask = offset < Size && (Data[offset++] & 1);
        if (use_memory_key_padding_mask && offset < Size) {
            memory_key_padding_mask = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Apply the TransformerDecoderLayer
        torch::Tensor output;
        
        if (use_tgt_mask && use_memory_mask && use_tgt_key_padding_mask && use_memory_key_padding_mask) {
            output = decoder_layer->forward(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask);
        } else if (use_tgt_mask && use_memory_mask && use_tgt_key_padding_mask) {
            output = decoder_layer->forward(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask);
        } else if (use_tgt_mask && use_memory_mask) {
            output = decoder_layer->forward(tgt, memory, tgt_mask, memory_mask);
        } else if (use_tgt_mask) {
            output = decoder_layer->forward(tgt, memory, tgt_mask);
        } else {
            output = decoder_layer->forward(tgt, memory);
        }
        
        // Ensure the output is used to prevent optimization
        if (output.defined()) {
            volatile auto sum = output.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
