#include "fuzzer_utils.h"
#include <iostream>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions for BatchNorm2d (N, C, H, W)
        if (input.dim() < 4) {
            while (input.dim() < 4) {
                input = input.unsqueeze(0);
            }
        } else if (input.dim() > 4) {
            // Reduce to 4D by flattening extra dimensions
            auto sizes = input.sizes().vec();
            int64_t batch = sizes[0];
            int64_t channels = sizes[1];
            int64_t height = sizes[2];
            int64_t remaining = 1;
            for (size_t i = 3; i < sizes.size(); i++) {
                remaining *= sizes[i];
            }
            input = input.reshape({batch, channels, height, remaining});
        }
        
        // Ensure channel dimension is at least 1
        int64_t num_features = input.size(1);
        if (num_features < 1) {
            return 0;
        }
        
        // Extract parameters for BatchNorm2d
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        if (offset + 4 <= Size) {
            uint32_t eps_raw = 0;
            std::memcpy(&eps_raw, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            // Keep eps in reasonable range
            eps = 1e-10 + (static_cast<double>(eps_raw) / std::numeric_limits<uint32_t>::max()) * 1e-3;
        }
        
        if (offset < Size) {
            momentum = static_cast<double>(Data[offset++]) / 255.0;
        }
        
        if (offset < Size) {
            affine = Data[offset++] % 2 == 0;
        }
        
        if (offset < Size) {
            track_running_stats = Data[offset++] % 2 == 0;
        }
        
        // Create BatchNorm2d module with num_features from input
        // Note: LazyBatchNorm2d is Python-only, use BatchNorm2d in C++
        torch::nn::BatchNorm2d bn(
            torch::nn::BatchNorm2dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Convert input to float for batch normalization
        input = input.to(torch::kFloat32);
        
        // Apply the module to the input tensor
        torch::Tensor output = bn->forward(input);
        
        // Test the module in training mode
        bn->train();
        torch::Tensor output_train = bn->forward(input);
        
        // Test in evaluation mode
        bn->eval();
        torch::Tensor output_eval = bn->forward(input);
        
        // Test with different input shapes (same channel count)
        if (offset + 2 < Size) {
            uint8_t height_byte = Data[offset++];
            uint8_t width_byte = Data[offset++];
            
            int64_t new_height = 1 + (height_byte % 32);
            int64_t new_width = 1 + (width_byte % 32);
            int64_t batch_size = std::max<int64_t>(1, input.size(0));
            
            try {
                torch::Tensor new_input = torch::randn({batch_size, num_features, new_height, new_width});
                torch::Tensor new_output = bn->forward(new_input);
            } catch (const std::exception& e) {
                // Silently ignore shape-related exceptions
            }
        }
        
        // Test reset_parameters if affine
        if (affine) {
            bn->reset_parameters();
            torch::Tensor output_after_reset = bn->forward(input);
        }
        
        // Test with batch size 1
        try {
            torch::Tensor single_batch = input.slice(0, 0, 1);
            bn->eval(); // Use eval mode for single batch
            torch::Tensor single_output = bn->forward(single_batch);
        } catch (const std::exception& e) {
            // Silently ignore
        }
        
        // Test running_mean and running_var access if tracking stats
        if (track_running_stats && bn->running_mean.defined()) {
            auto mean = bn->running_mean;
            auto var = bn->running_var;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}