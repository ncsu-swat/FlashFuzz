#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Early return if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple Conv2d + BatchNorm2d + ReLU combination
        int64_t in_channels = 3;
        int64_t out_channels = 3;
        int64_t kernel_size = 3;
        
        // Adjust dimensions if needed based on input tensor
        if (input.dim() >= 3) {
            in_channels = input.size(1) > 0 ? input.size(1) : 3;
        }
        
        // Create the modules
        auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                        .stride(1)
                                        .padding(1)
                                        .dilation(1)
                                        .groups(1)
                                        .bias(true));
        
        auto bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels));
        auto relu = torch::nn::ReLU();
        
        // Get a byte from the input data to determine whether to update BN stats
        bool update_bn_stats = true;
        if (offset < Size) {
            update_bn_stats = Data[offset++] % 2 == 0;
        }
        
        // Set the modules to training mode
        conv->train();
        bn->train();
        
        // Ensure input has the right shape for Conv2d (N, C, H, W)
        if (input.dim() < 4) {
            // Reshape to a valid 4D tensor
            std::vector<int64_t> new_shape = {1, in_channels, 5, 5};
            input = input.reshape(new_shape);
        } else if (input.size(1) != in_channels) {
            // Adjust the channel dimension
            std::vector<int64_t> new_shape = input.sizes().vec();
            new_shape[1] = in_channels;
            input = input.reshape(new_shape);
        }
        
        // Convert to float if needed
        if (input.scalar_type() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Apply the operations
        torch::Tensor output;
        
        // Forward pass through conv -> bn -> relu
        auto conv_out = conv->forward(input);
        
        // Test BatchNorm with different track_running_stats settings
        if (update_bn_stats) {
            bn->options.track_running_stats(true);
        } else {
            bn->options.track_running_stats(false);
        }
        
        auto bn_out = bn->forward(conv_out);
        output = relu->forward(bn_out);
        
        // Test with opposite flag value
        bn->options.track_running_stats(!update_bn_stats);
        bn_out = bn->forward(conv_out);
        output = relu->forward(bn_out);
        
        // Test with eval mode (freezes BN stats)
        bn->eval();
        bn_out = bn->forward(conv_out);
        output = relu->forward(bn_out);
        
        // Test with train mode (updates BN stats)
        bn->train();
        bn_out = bn->forward(conv_out);
        output = relu->forward(bn_out);
        
        // Try to access and manipulate BN parameters
        if (bn->options.momentum().has_value()) {
            bn->options.momentum(0.5);
        }
        if (bn->options.eps().has_value()) {
            bn->options.eps(1e-5);
        }
        bn_out = bn->forward(conv_out);
        output = relu->forward(bn_out);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
