#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    if (Size < 12) {
        return 0;
    }

    try
    {
        size_t offset = 0;

        // Parse EmbeddingBag parameters first (need num_embeddings for indices)
        int64_t num_embeddings = static_cast<int64_t>(Data[offset++] % 100) + 1;
        int64_t embedding_dim = static_cast<int64_t>(Data[offset++] % 64) + 1;

        // Parse mode: 0=sum, 1=mean, 2=max
        int64_t mode = Data[offset++] % 3;

        // Parse flags
        bool sparse = Data[offset++] % 2 == 0;
        bool scale_grad_by_freq = Data[offset++] % 2 == 0;
        bool include_last_offset = Data[offset++] % 2 == 0;

        // Parse padding_idx
        int64_t padding_idx = -1;
        if (Data[offset++] % 3 != 0) {
            padding_idx = Data[offset - 1] % num_embeddings;
        }

        // Parse indices size and number of bags
        int64_t num_indices = static_cast<int64_t>(Data[offset++] % 50) + 1;
        int64_t num_bags = static_cast<int64_t>(Data[offset++] % 10) + 1;

        // Parse whether to use per-sample weights
        // max mode doesn't support weights
        bool use_weights = (Data[offset++] % 2 == 0) && (mode != 2);

        // Create indices tensor with valid values in [0, num_embeddings)
        std::vector<int64_t> indices_data(num_indices);
        for (int64_t i = 0; i < num_indices; i++) {
            if (offset < Size) {
                indices_data[i] = Data[offset++] % num_embeddings;
            } else {
                indices_data[i] = i % num_embeddings;
            }
        }
        torch::Tensor indices = torch::tensor(indices_data, torch::kLong);

        // Create offsets tensor (must be sorted, start at 0)
        int64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
        std::vector<int64_t> offsets_data(offsets_size);
        offsets_data[0] = 0;
        for (int64_t i = 1; i < offsets_size; i++) {
            // Each offset should be >= previous and <= num_indices
            int64_t remaining = num_indices - offsets_data[i - 1];
            int64_t remaining_slots = offsets_size - i;
            int64_t max_step = remaining_slots > 0 ? remaining / remaining_slots : 0;
            if (offset < Size && max_step > 0) {
                offsets_data[i] = offsets_data[i - 1] + (Data[offset++] % (max_step + 1));
            } else {
                offsets_data[i] = offsets_data[i - 1];
            }
        }
        // Last offset must equal num_indices when include_last_offset is true
        if (include_last_offset) {
            offsets_data[offsets_size - 1] = num_indices;
        }
        torch::Tensor offsets = torch::tensor(offsets_data, torch::kLong);

        // Create per-sample weights if needed (must match indices size)
        torch::Tensor per_sample_weights;
        if (use_weights) {
            per_sample_weights = torch::randn({num_indices}, torch::kFloat);
        }

        // Create EmbeddingBag options
        torch::nn::EmbeddingBagOptions options(num_embeddings, embedding_dim);

        if (mode == 0) {
            options.mode(torch::kSum);
        } else if (mode == 1) {
            options.mode(torch::kMean);
        } else {
            options.mode(torch::kMax);
        }

        options.sparse(sparse);
        options.scale_grad_by_freq(scale_grad_by_freq);
        options.include_last_offset(include_last_offset);

        if (padding_idx >= 0) {
            options.padding_idx(padding_idx);
        }

        // Create EmbeddingBag module
        torch::nn::EmbeddingBag embeddingBag(options);

        // Forward pass with 1D indices and offsets
        torch::Tensor output;
        try {
            if (use_weights && per_sample_weights.defined()) {
                output = embeddingBag->forward(indices, offsets, per_sample_weights);
            } else {
                output = embeddingBag->forward(indices, offsets);
            }
            
            // Verify output shape
            if (output.defined()) {
                (void)output.sizes();
            }
        } catch (const c10::Error&) {
            // Expected failures for invalid configurations
        } catch (const std::runtime_error&) {
            // Expected failures
        }

        // Access weight parameter
        try {
            auto weight = embeddingBag->weight;
            if (weight.defined()) {
                (void)weight.sizes();
            }
        } catch (...) {
            // Ignore weight access issues
        }

        // Test with 2D indices (no offsets needed)
        if (offset + 2 < Size) {
            int64_t batch_size = Data[offset++] % 8 + 1;
            int64_t seq_len = Data[offset++] % 10 + 1;

            std::vector<int64_t> indices_2d_data(batch_size * seq_len);
            for (size_t i = 0; i < indices_2d_data.size(); i++) {
                if (offset < Size) {
                    indices_2d_data[i] = Data[offset++] % num_embeddings;
                } else {
                    indices_2d_data[i] = i % num_embeddings;
                }
            }
            torch::Tensor indices_2d = torch::tensor(indices_2d_data, torch::kLong).reshape({batch_size, seq_len});

            try {
                // 2D input doesn't need offsets
                output = embeddingBag->forward(indices_2d);
                if (output.defined()) {
                    (void)output.sizes();
                }
            } catch (const c10::Error&) {
                // Expected failures
            } catch (const std::runtime_error&) {
                // Expected failures
            }
        }

        // Test from_pretrained static method
        if (offset + 1 < Size && Data[offset++] % 4 == 0) {
            torch::Tensor pretrained = torch::randn({num_embeddings, embedding_dim});
            try {
                torch::nn::EmbeddingBagOptions from_opts(num_embeddings, embedding_dim);
                from_opts.include_last_offset(include_last_offset);
                if (mode == 0) {
                    from_opts.mode(torch::kSum);
                } else if (mode == 1) {
                    from_opts.mode(torch::kMean);
                } else {
                    from_opts.mode(torch::kMax);
                }
                
                auto embeddingBag2 = torch::nn::EmbeddingBag(from_opts);
                // Copy pretrained weights
                torch::NoGradGuard no_grad;
                embeddingBag2->weight.copy_(pretrained);
                
                output = embeddingBag2->forward(indices, offsets);
                if (output.defined()) {
                    (void)output.sizes();
                }
            } catch (const c10::Error&) {
                // Expected failures
            } catch (const std::runtime_error&) {
                // Expected failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}