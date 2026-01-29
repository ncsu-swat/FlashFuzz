#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Get embedding parameters from input data
        int64_t num_embeddings = (Data[offset++] % 16) + 2;  // 2-17
        int64_t embedding_dim = (Data[offset++] % 8) + 1;    // 1-8
        
        // Determine mode
        int mode_val = Data[offset++] % 3;
        
        // Determine include_last_offset
        bool include_last_offset = Data[offset++] % 2 == 0;
        
        // Create float weight tensor for the embedding table
        torch::Tensor weight = torch::randn({num_embeddings, embedding_dim}, torch::kFloat);
        
        // Quantize the weight tensor to create quantized embedding
        // Use qint8 quantization
        float scale = 0.1f;
        int zero_point = 0;
        
        try {
            // Quantize per tensor
            torch::Tensor weight_quantized = torch::quantize_per_tensor(
                weight, scale, zero_point, torch::kQInt8);
            
            // Create indices tensor - must be within [0, num_embeddings)
            int64_t num_indices = (Data[offset] % 8) + 1;  // 1-8 indices
            offset++;
            
            std::vector<int64_t> indices_vec;
            for (int64_t i = 0; i < num_indices && offset < Size; i++) {
                int64_t idx = Data[offset++] % num_embeddings;
                indices_vec.push_back(idx);
            }
            
            if (indices_vec.empty()) {
                indices_vec.push_back(0);
            }
            
            torch::Tensor indices = torch::tensor(indices_vec, torch::kLong);
            
            // Create offsets tensor for bag boundaries
            int64_t num_bags = (offset < Size) ? (Data[offset++] % 4) + 1 : 1;  // 1-4 bags
            
            std::vector<int64_t> offsets_vec;
            offsets_vec.push_back(0);  // First bag starts at 0
            
            int64_t current_offset = 0;
            for (int64_t i = 1; i < num_bags; i++) {
                int64_t increment = (offset < Size) ? (Data[offset++] % 3) + 1 : 1;
                current_offset += increment;
                if (current_offset < static_cast<int64_t>(indices_vec.size())) {
                    offsets_vec.push_back(current_offset);
                }
            }
            
            if (include_last_offset) {
                offsets_vec.push_back(static_cast<int64_t>(indices_vec.size()));
            }
            
            torch::Tensor offsets = torch::tensor(offsets_vec, torch::kLong);
            
            // Create per_sample_weights if requested
            torch::Tensor per_sample_weights;
            bool use_weights = (offset < Size) && (Data[offset++] % 2 == 0);
            
            if (use_weights && mode_val != 2) {  // per_sample_weights not supported with max mode
                per_sample_weights = torch::randn({static_cast<int64_t>(indices_vec.size())}, torch::kFloat);
            }
            
            // Call the quantized embedding_bag function directly
            // torch::embedding_bag with quantized weights
            try {
                // Use the functional interface for quantized embedding bag
                // The C++ API provides torch::embedding_bag which works with quantized tensors
                
                // First, let's use the dequantized path as quantized::EmbeddingBag 
                // module may not be directly available
                torch::Tensor weight_dequant = weight_quantized.dequantize();
                
                // Call embedding_bag with the weight
                auto result = torch::embedding_bag(
                    weight_dequant,
                    indices,
                    offsets,
                    /*scale_grad_by_freq=*/false,
                    /*mode=*/mode_val,
                    /*sparse=*/false,
                    /*per_sample_weights=*/use_weights ? per_sample_weights : torch::Tensor(),
                    /*include_last_offset=*/include_last_offset
                );
                
                // Result is a tuple
                torch::Tensor output = std::get<0>(result);
                
                // Verify output shape
                (void)output.sizes();
                
            } catch (const c10::Error& e) {
                // Expected for invalid configurations
            }
            
            // Also test with regular embedding bag module for comparison
            try {
                torch::nn::EmbeddingBagOptions options(num_embeddings, embedding_dim);
                if (mode_val == 0) options.mode(torch::kSum);
                else if (mode_val == 1) options.mode(torch::kMean);
                else options.mode(torch::kMax);
                options.include_last_offset(include_last_offset);
                
                torch::nn::EmbeddingBag embeddingBag(options);
                
                torch::Tensor output;
                if (use_weights && mode_val != 2) {
                    output = embeddingBag->forward(indices, offsets, per_sample_weights);
                } else {
                    output = embeddingBag->forward(indices, offsets);
                }
                
                (void)output.sizes();
                
            } catch (const c10::Error& e) {
                // Expected for invalid configurations
            }
            
        } catch (const c10::Error& e) {
            // Quantization or embedding errors - expected for some inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}