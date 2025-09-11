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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create indices tensor
        torch::Tensor indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // If we have more data, create weights tensor
        torch::Tensor weights;
        if (offset < Size) {
            weights = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Extract parameters for Embedding
        int64_t num_embeddings = 10;
        int64_t embedding_dim = 3;
        
        // If we have more data, use it to set parameters
        if (offset + 16 <= Size) {
            int64_t raw_num_embeddings, raw_embedding_dim;
            std::memcpy(&raw_num_embeddings, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&raw_embedding_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure values are reasonable but allow edge cases
            num_embeddings = std::abs(raw_num_embeddings) % 1000 + 1;
            embedding_dim = std::abs(raw_embedding_dim) % 100 + 1;
        }
        
        // Extract additional parameters
        bool sparse = false;
        double padding_idx = -1;
        double max_norm = 0.0;
        double norm_type = 2.0;
        bool scale_grad_by_freq = false;
        
        if (offset < Size) {
            sparse = Data[offset++] & 0x1;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&padding_idx, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_norm, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&norm_type, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset < Size) {
            scale_grad_by_freq = Data[offset++] & 0x1;
        }
        
        // Create regular Embedding module (QAT not available in C++ frontend)
        torch::nn::EmbeddingOptions options = torch::nn::EmbeddingOptions(num_embeddings, embedding_dim)
            .sparse(sparse)
            .max_norm(max_norm)
            .norm_type(norm_type)
            .scale_grad_by_freq(scale_grad_by_freq);
        
        // Handle padding_idx specially to allow for None
        if (padding_idx >= 0 && padding_idx < num_embeddings) {
            options = options.padding_idx(static_cast<int64_t>(padding_idx));
        }
        
        auto embedding = torch::nn::Embedding(options);
        
        // Forward pass
        torch::Tensor output;
        if (weights.defined() && !weights.sizes().empty()) {
            output = embedding->forward(indices, weights);
        } else {
            output = embedding->forward(indices);
        }
        
        // Test with quantization operations
        torch::Tensor quantized_output = torch::quantize_per_tensor(output, 0.1, 128, torch::kQUInt8);
        torch::Tensor dequantized_output = torch::dequantize(quantized_output);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
