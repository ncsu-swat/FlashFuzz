#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>
#include <variant>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need enough bytes to create meaningful tensors
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract batch size and embedding dimension from data
        uint8_t batch_byte = Data[offset++];
        uint8_t dim_byte = Data[offset++];
        
        int64_t batch_size = (batch_byte % 16) + 1;  // 1-16
        int64_t embedding_dim = (dim_byte % 64) + 1; // 1-64

        // Create input tensors with controlled shapes (N, D)
        torch::Tensor input1 = torch::randn({batch_size, embedding_dim});
        torch::Tensor input2 = torch::randn({batch_size, embedding_dim});

        // Seed the random tensors with fuzzer data if available
        if (offset + 4 <= Size) {
            int32_t seed;
            std::memcpy(&seed, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            torch::manual_seed(seed);
            input1 = torch::randn({batch_size, embedding_dim});
            input2 = torch::randn({batch_size, embedding_dim});
        }

        // Create target tensor (1D with values of 1 or -1)
        torch::Tensor target = torch::ones({batch_size});
        if (offset < Size) {
            // Use fuzzer data to determine which targets are -1
            for (int64_t i = 0; i < batch_size && offset < Size; i++) {
                if (Data[offset++] % 2 == 0) {
                    target[i] = -1;
                }
            }
        }

        // Get reduction mode from the data
        // Track which reduction mode we're using for later
        int reduction_selector = 2; // default to mean
        if (offset < Size) {
            reduction_selector = Data[offset++] % 3;
        }

        torch::nn::CosineEmbeddingLossOptions::reduction_t reduction_mode;
        switch (reduction_selector) {
            case 0:
                reduction_mode = torch::kNone;
                break;
            case 1:
                reduction_mode = torch::kSum;
                break;
            default:
                reduction_mode = torch::kMean;
                break;
        }

        // Get margin value from the data (clamp to valid range)
        double margin = 0.0;
        if (offset + sizeof(float) <= Size) {
            float margin_f;
            std::memcpy(&margin_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Clamp margin to a reasonable range and handle NaN/Inf
            if (std::isfinite(margin_f)) {
                margin = std::max(-1.0, std::min(1.0, static_cast<double>(margin_f)));
            }
        }

        // Create CosineEmbeddingLoss module
        auto options = torch::nn::CosineEmbeddingLossOptions()
                          .margin(margin)
                          .reduction(reduction_mode);

        torch::nn::CosineEmbeddingLoss loss_fn(options);

        // Apply the loss function
        torch::Tensor loss = loss_fn(input1, input2, target);

        // Force evaluation
        // Use the selector we tracked instead of comparing variants
        if (reduction_selector == 0) {
            // Result is a tensor, access first element
            volatile float loss_value = loss[0].item<float>();
            (void)loss_value;
        } else {
            // Result is scalar
            volatile float loss_value = loss.item<float>();
            (void)loss_value;
        }

        // Also test with 1D inputs (single sample case)
        if (offset < Size && (Data[offset] % 4 == 0)) {
            torch::Tensor input1_1d = input1[0];
            torch::Tensor input2_1d = input2[0];
            torch::Tensor target_1d = target[0].unsqueeze(0);
            
            try {
                torch::Tensor loss_1d = loss_fn(input1_1d.unsqueeze(0), input2_1d.unsqueeze(0), target_1d);
                volatile float val = loss_1d.item<float>();
                (void)val;
            } catch (...) {
                // Expected for some configurations, ignore silently
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