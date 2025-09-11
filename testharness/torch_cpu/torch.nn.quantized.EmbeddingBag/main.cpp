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
        
        // Parse parameters for EmbeddingBag
        int64_t num_embeddings = 0;
        int64_t embedding_dim = 0;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&num_embeddings, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_embeddings = std::abs(num_embeddings) % 100 + 1;
        } else {
            num_embeddings = 10;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&embedding_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            embedding_dim = std::abs(embedding_dim) % 64 + 1;
        } else {
            embedding_dim = 8;
        }
        
        // Parse mode
        int64_t mode_raw = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&mode_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        mode_raw = std::abs(mode_raw) % 3;
        
        // Parse sparse flag
        bool sparse = false;
        if (offset < Size) {
            sparse = Data[offset++] & 0x1;
        }
        
        // Parse include_last_offset flag
        bool include_last_offset = false;
        if (offset < Size) {
            include_last_offset = Data[offset++] & 0x1;
        }
        
        // Parse scale and zero_point for quantization
        float scale = 1.0f;
        int32_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (!std::isfinite(scale) || scale == 0.0f) {
                scale = 1.0f;
            }
            scale = std::abs(scale);
        }
        
        if (offset + sizeof(int32_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            zero_point = zero_point % 256;
        }
        
        // Create weight tensor for EmbeddingBag
        std::vector<int64_t> weight_shape = {num_embeddings, embedding_dim};
        auto weight_options = torch::TensorOptions().dtype(torch::kFloat);
        auto weight = torch::rand(weight_shape, weight_options);
        
        // Quantize the weight tensor
        auto quantized_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQUInt8);
        
        // Create indices tensor
        torch::Tensor indices;
        if (offset < Size) {
            try {
                indices = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure indices are valid (non-negative and less than num_embeddings)
                indices = torch::abs(indices) % num_embeddings;
                
                // Convert to int64 if not already
                if (indices.dtype() != torch::kLong) {
                    indices = indices.to(torch::kLong);
                }
            } catch (const std::exception&) {
                // If tensor creation fails, create a simple indices tensor
                indices = torch::randint(0, num_embeddings, {5}, torch::kLong);
            }
        } else {
            indices = torch::randint(0, num_embeddings, {5}, torch::kLong);
        }
        
        // Create offsets tensor for mode=0 (bag)
        torch::Tensor offsets;
        if (mode_raw == 0) {
            try {
                if (offset < Size) {
                    offsets = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Ensure offsets are valid (sorted and within indices size)
                    if (offsets.dtype() != torch::kLong) {
                        offsets = offsets.to(torch::kLong);
                    }
                    
                    // Sort offsets and ensure they're non-negative
                    offsets = torch::abs(offsets);
                    offsets = std::get<0>(torch::sort(offsets));
                    
                    // Ensure offsets are within bounds of indices
                    offsets = offsets % (indices.size(0) + 1);
                }
                else {
                    // Create simple offsets if not enough data
                    offsets = torch::tensor({0, 2, 5}, torch::kLong);
                }
            } catch (const std::exception&) {
                // Create simple offsets if tensor creation fails
                offsets = torch::tensor({0, 2, 5}, torch::kLong);
            }
        }
        
        // Create per_sample_weights tensor (optional)
        torch::Tensor per_sample_weights;
        bool use_per_sample_weights = false;
        if (offset < Size) {
            use_per_sample_weights = Data[offset++] & 0x1;
            
            if (use_per_sample_weights) {
                try {
                    if (offset < Size) {
                        per_sample_weights = fuzzer_utils::createTensor(Data, Size, offset);
                        
                        // Ensure per_sample_weights has same size as indices
                        if (per_sample_weights.size(0) != indices.size(0)) {
                            per_sample_weights = torch::ones({indices.size(0)}, torch::kFloat);
                        }
                        
                        // Convert to float if not already
                        if (per_sample_weights.dtype() != torch::kFloat) {
                            per_sample_weights = per_sample_weights.to(torch::kFloat);
                        }
                    }
                } catch (const std::exception&) {
                    per_sample_weights = torch::ones({indices.size(0)}, torch::kFloat);
                }
            }
        }
        
        // Convert mode_raw to proper EmbeddingBagMode
        torch::nn::EmbeddingBagMode mode;
        switch (mode_raw) {
            case 0:
                mode = torch::kSum;
                break;
            case 1:
                mode = torch::kMean;
                break;
            case 2:
                mode = torch::kMax;
                break;
            default:
                mode = torch::kSum;
                break;
        }
        
        // Create the EmbeddingBag module options
        torch::nn::EmbeddingBagOptions options(num_embeddings, embedding_dim);
        options = options.mode(mode).sparse(sparse).include_last_offset(include_last_offset);
        
        auto embedding_bag = torch::nn::EmbeddingBag(options);
        
        // Set the quantized weights using the weight parameter
        embedding_bag->weight = quantized_weight;
        
        // Forward pass
        torch::Tensor output;
        if (mode_raw == 0) {
            // Mode: sum - requires offsets
            if (use_per_sample_weights) {
                output = embedding_bag->forward(indices, offsets, per_sample_weights);
            } else {
                output = embedding_bag->forward(indices, offsets);
            }
        } else {
            // Mode: mean or max - doesn't require offsets
            if (use_per_sample_weights) {
                output = embedding_bag->forward(indices, torch::Tensor(), per_sample_weights);
            } else {
                output = embedding_bag->forward(indices);
            }
        }
        
        // Verify output shape
        int64_t expected_batch_size = (mode_raw == 0) ? offsets.size(0) - (include_last_offset ? 1 : 0) : 1;
        if (output.size(0) != expected_batch_size || output.size(1) != embedding_dim) {
            throw std::runtime_error("Output shape mismatch");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
