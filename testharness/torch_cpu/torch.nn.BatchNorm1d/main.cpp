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

    try
    {
        // Need at least a few bytes for parameters
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters from fuzzer data
        int64_t num_features = (Data[offset++] % 64) + 1;  // 1-64 features
        int64_t batch_size = (Data[offset++] % 16) + 1;    // 1-16 batch size
        int64_t seq_len = Data[offset++] % 32;             // 0-31 sequence length (0 means 2D input)
        
        // Extract BatchNorm parameters
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + sizeof(float) <= Size) {
            float eps_f;
            std::memcpy(&eps_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            eps = std::abs(eps_f);
            if (eps < 1e-10 || std::isnan(eps) || std::isinf(eps)) {
                eps = 1e-5;
            }
        }
        
        if (offset + sizeof(float) <= Size) {
            float mom_f;
            std::memcpy(&mom_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (!std::isnan(mom_f) && !std::isinf(mom_f)) {
                momentum = std::max(0.0, std::min(1.0, static_cast<double>(mom_f)));
            }
        }
        
        bool affine = (offset < Size) ? (Data[offset++] % 2) == 1 : true;
        bool track_running_stats = (offset < Size) ? (Data[offset++] % 2) == 1 : true;
        
        // Create input tensor with appropriate shape for BatchNorm1d
        // BatchNorm1d expects (N, C) or (N, C, L)
        torch::Tensor input;
        if (seq_len > 0) {
            // 3D input: (N, C, L)
            input = torch::randn({batch_size, num_features, seq_len});
        } else {
            // 2D input: (N, C)
            input = torch::randn({batch_size, num_features});
        }
        
        // Use remaining data to perturb input values
        if (offset < Size) {
            torch::Tensor noise = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                // Try to add noise if shapes are compatible
                if (noise.numel() > 0) {
                    auto flat_noise = noise.flatten().slice(0, 0, std::min(noise.numel(), input.numel()));
                    auto flat_input = input.flatten();
                    flat_input.slice(0, 0, flat_noise.numel()).add_(flat_noise.to(torch::kFloat) * 0.1f);
                }
            } catch (...) {
                // Ignore noise addition failures
            }
        }
        
        // Create BatchNorm1d module
        torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(num_features)
                                  .eps(eps)
                                  .momentum(momentum)
                                  .affine(affine)
                                  .track_running_stats(track_running_stats));
        
        // Test in training mode (default)
        bn->train();
        torch::Tensor train_output = bn->forward(input);
        auto train_sum = train_output.sum();
        
        // Test in eval mode
        bn->eval();
        torch::Tensor eval_output = bn->forward(input);
        auto eval_sum = eval_output.sum();
        
        // Test with different batch sizes (only in training mode with track_running_stats)
        if (batch_size > 1) {
            try {
                bn->train();
                torch::Tensor smaller_batch = input.slice(0, 0, batch_size / 2 + 1);
                torch::Tensor smaller_output = bn->forward(smaller_batch);
            } catch (...) {
                // May fail with batch size 1 in training mode
            }
        }
        
        // Test double precision
        try {
            auto bn_double = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(num_features)
                                                    .eps(eps)
                                                    .momentum(momentum)
                                                    .affine(affine)
                                                    .track_running_stats(track_running_stats));
            bn_double->to(torch::kDouble);
            torch::Tensor input_double = input.to(torch::kDouble);
            bn_double->eval();
            torch::Tensor output_double = bn_double->forward(input_double);
        } catch (...) {
            // Ignore dtype conversion failures
        }
        
        // Test reset_running_stats if tracking
        if (track_running_stats) {
            bn->reset_running_stats();
            bn->train();
            torch::Tensor output_after_reset = bn->forward(input);
        }
        
        // Test with batch size 1 in eval mode
        try {
            bn->eval();
            torch::Tensor single_sample;
            if (seq_len > 0) {
                single_sample = torch::randn({1, num_features, seq_len});
            } else {
                single_sample = torch::randn({1, num_features});
            }
            torch::Tensor single_output = bn->forward(single_sample);
        } catch (...) {
            // May fail if running stats not initialized
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}