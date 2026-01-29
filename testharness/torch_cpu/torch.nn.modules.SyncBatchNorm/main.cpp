#include "fuzzer_utils.h"
#include <iostream>
#include <sstream>

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
        
        // Extract parameters from fuzzer data
        double momentum = static_cast<double>(Data[offset++]) / 255.0;
        double eps = std::max(1e-10, static_cast<double>(Data[offset++]) / 1e4);
        bool affine = Data[offset++] % 2 == 0;
        bool track_running_stats = Data[offset++] % 2 == 0;
        uint8_t batch_size = std::max(1, static_cast<int>(Data[offset++] % 8) + 1);
        uint8_t num_channels = std::max(1, static_cast<int>(Data[offset++] % 16) + 1);
        uint8_t spatial_dim = std::max(1, static_cast<int>(Data[offset++] % 8) + 1);
        uint8_t mode = Data[offset++] % 3; // 0: 1d, 1: 2d, 2: 3d
        
        torch::Tensor input;
        
        // Create input tensor with appropriate dimensions for different BatchNorm variants
        if (mode == 0) {
            // BatchNorm1d: (N, C) or (N, C, L)
            if (offset < Size && Data[offset] % 2 == 0) {
                input = torch::randn({batch_size, num_channels});
            } else {
                input = torch::randn({batch_size, num_channels, spatial_dim});
            }
            offset++;
        } else if (mode == 1) {
            // BatchNorm2d: (N, C, H, W)
            input = torch::randn({batch_size, num_channels, spatial_dim, spatial_dim});
        } else {
            // BatchNorm3d: (N, C, D, H, W)
            input = torch::randn({batch_size, num_channels, spatial_dim, spatial_dim, spatial_dim});
        }
        
        // Use remaining fuzzer data to perturb input values
        if (offset < Size) {
            torch::Tensor noise = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                // Try to add noise if shapes are compatible
                if (noise.numel() > 0) {
                    noise = noise.flatten().slice(0, 0, std::min(noise.numel(), input.numel()));
                    noise = noise.view({-1}).expand({input.numel()}).view(input.sizes()).to(input.dtype());
                    input = input + noise * 0.01;
                }
            } catch (...) {
                // Shape mismatch is fine, continue with original input
            }
        }
        
        if (mode == 0) {
            // Test SyncBatchNorm with 1D-like input
            // SyncBatchNorm in C++ API - without distributed setup, it behaves like BatchNorm
            auto sync_bn = torch::nn::SyncBatchNorm(
                torch::nn::SyncBatchNormOptions(num_channels)
                    .eps(eps)
                    .momentum(momentum)
                    .affine(affine)
                    .track_running_stats(track_running_stats)
            );
            
            // Forward pass in train mode
            sync_bn->train();
            torch::Tensor output_train = sync_bn->forward(input);
            
            // Forward pass in eval mode
            sync_bn->eval();
            torch::Tensor output_eval = sync_bn->forward(input);
            
            // Test parameter access if affine
            if (affine) {
                auto weight = sync_bn->weight;
                auto bias = sync_bn->bias;
                (void)weight;
                (void)bias;
            }
            
            // Test running stats access if tracking
            if (track_running_stats) {
                auto running_mean = sync_bn->running_mean;
                auto running_var = sync_bn->running_var;
                auto num_batches = sync_bn->num_batches_tracked;
                (void)running_mean;
                (void)running_var;
                (void)num_batches;
            }
            
            // Test serialization
            std::stringstream ss;
            torch::serialize::OutputArchive out_archive;
            sync_bn->save(out_archive);
            out_archive.save_to(ss);
            
            torch::serialize::InputArchive in_archive;
            in_archive.load_from(ss);
            auto loaded_bn = torch::nn::SyncBatchNorm(
                torch::nn::SyncBatchNormOptions(num_channels)
            );
            loaded_bn->load(in_archive);
            
            torch::Tensor loaded_output = loaded_bn->forward(input);
            
        } else if (mode == 1) {
            // Test SyncBatchNorm with 2D-like input (4D tensor)
            auto sync_bn = torch::nn::SyncBatchNorm(
                torch::nn::SyncBatchNormOptions(num_channels)
                    .eps(eps)
                    .momentum(momentum)
                    .affine(affine)
                    .track_running_stats(track_running_stats)
            );
            
            sync_bn->train();
            torch::Tensor output = sync_bn->forward(input);
            
            sync_bn->eval();
            torch::Tensor eval_output = sync_bn->forward(input);
            
        } else {
            // Test SyncBatchNorm with 3D-like input (5D tensor)
            auto sync_bn = torch::nn::SyncBatchNorm(
                torch::nn::SyncBatchNormOptions(num_channels)
                    .eps(eps)
                    .momentum(momentum)
                    .affine(affine)
                    .track_running_stats(track_running_stats)
            );
            
            sync_bn->train();
            torch::Tensor output = sync_bn->forward(input);
            
            sync_bn->eval();
            torch::Tensor eval_output = sync_bn->forward(input);
        }
        
        // Test convert_sync_batchnorm static method if available
        // This converts BatchNorm modules to SyncBatchNorm
        auto regular_bn = torch::nn::BatchNorm1d(
            torch::nn::BatchNorm1dOptions(num_channels)
        );
        
        // Clone the regular batchnorm for testing
        auto cloned = regular_bn->clone();
        (void)cloned;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}