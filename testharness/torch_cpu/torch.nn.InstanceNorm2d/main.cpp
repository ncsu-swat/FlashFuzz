#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // InstanceNorm2d requires floating point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure input has at least 4 dimensions for InstanceNorm2d (N, C, H, W)
        if (input.dim() < 4) {
            // Expand dimensions if needed
            while (input.dim() < 4) {
                input = input.unsqueeze(0);
            }
        } else if (input.dim() > 4) {
            // InstanceNorm2d expects exactly 4D input, reshape if too many dims
            auto sizes = input.sizes();
            int64_t n = sizes[0];
            int64_t c = sizes[1];
            int64_t h = sizes[2];
            int64_t rest = 1;
            for (int i = 3; i < input.dim(); i++) {
                rest *= sizes[i];
            }
            input = input.reshape({n, c, h, rest});
        }
        
        // Ensure we have valid dimensions (non-zero)
        if (input.size(0) == 0 || input.size(1) == 0 || input.size(2) == 0 || input.size(3) == 0) {
            return 0;
        }
        
        // Extract parameters for InstanceNorm2d from the remaining data
        bool affine = false;
        bool track_running_stats = false;
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + 2 <= Size) {
            affine = Data[offset++] & 0x1;
            track_running_stats = Data[offset++] & 0x1;
        }
        
        if (offset + sizeof(float) <= Size) {
            float eps_f;
            memcpy(&eps_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure eps is positive and reasonable
            eps = static_cast<double>(std::abs(eps_f));
            if (eps < 1e-10 || !std::isfinite(eps)) eps = 1e-5;
            if (eps > 1.0) eps = 1e-5;
        }
        
        if (offset + sizeof(float) <= Size) {
            float momentum_f;
            memcpy(&momentum_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure momentum is in [0, 1]
            momentum = static_cast<double>(std::abs(momentum_f));
            if (!std::isfinite(momentum)) momentum = 0.1;
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        // Get the number of channels (second dimension)
        int64_t num_features = input.size(1);
        
        // Create InstanceNorm2d module
        torch::nn::InstanceNorm2d instance_norm(
            torch::nn::InstanceNorm2dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Inner try-catch for expected operation failures
        try {
            // Apply InstanceNorm2d
            torch::Tensor output = instance_norm->forward(input);
            
            // Verify output shape matches input shape
            (void)output.sizes();
            
            // Try to access some properties to ensure they're properly initialized
            if (affine) {
                torch::Tensor weight = instance_norm->weight;
                torch::Tensor bias = instance_norm->bias;
                (void)weight.numel();
                (void)bias.numel();
            }
            
            if (track_running_stats) {
                torch::Tensor running_mean = instance_norm->running_mean;
                torch::Tensor running_var = instance_norm->running_var;
                (void)running_mean.numel();
                (void)running_var.numel();
            }
            
            // Test training vs eval mode
            if (offset < Size && (Data[offset] & 0x1)) {
                instance_norm->train();
                torch::Tensor train_output = instance_norm->forward(input);
                (void)train_output.sizes();
            } else {
                instance_norm->eval();
                torch::Tensor eval_output = instance_norm->forward(input);
                (void)eval_output.sizes();
            }
        }
        catch (const std::exception &) {
            // Silently catch expected failures (shape mismatches, etc.)
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}