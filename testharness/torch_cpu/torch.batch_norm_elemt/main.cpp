#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

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
        
        // Need at least some data to proceed
        if (Size < 5) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // batch_norm_elemt expects input with at least 2 dimensions (N, C, ...)
        // If input has fewer dimensions, reshape or skip
        if (input.dim() < 2) {
            // Reshape to at least 2D: treat as (1, C) where C is the total size
            int64_t total_size = input.numel();
            if (total_size == 0) {
                return 0;
            }
            input = input.reshape({1, total_size});
        }
        
        // Get number of channels from dimension 1
        int64_t num_channels = input.size(1);
        if (num_channels == 0) {
            return 0;
        }
        
        // Get the dtype of the input tensor to ensure consistency
        auto dtype = input.dtype();
        
        // Create weight tensor (1D with size = num_channels)
        torch::Tensor weight;
        if (offset < Size) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
            // Flatten and resize to match num_channels
            weight = weight.flatten();
            if (weight.numel() >= num_channels) {
                weight = weight.slice(0, 0, num_channels).to(dtype);
            } else {
                weight = torch::ones({num_channels}, dtype);
            }
        } else {
            weight = torch::ones({num_channels}, dtype);
        }
        
        // Create bias tensor (1D with size = num_channels)
        torch::Tensor bias;
        if (offset < Size) {
            bias = fuzzer_utils::createTensor(Data, Size, offset);
            bias = bias.flatten();
            if (bias.numel() >= num_channels) {
                bias = bias.slice(0, 0, num_channels).to(dtype);
            } else {
                bias = torch::zeros({num_channels}, dtype);
            }
        } else {
            bias = torch::zeros({num_channels}, dtype);
        }
        
        // Create mean tensor (1D with size = num_channels)
        torch::Tensor mean;
        if (offset < Size) {
            mean = fuzzer_utils::createTensor(Data, Size, offset);
            mean = mean.flatten();
            if (mean.numel() >= num_channels) {
                mean = mean.slice(0, 0, num_channels).to(dtype);
            } else {
                mean = torch::zeros({num_channels}, dtype);
            }
        } else {
            mean = torch::zeros({num_channels}, dtype);
        }
        
        // Create invstd tensor (inverse standard deviation, 1D with size = num_channels)
        // Note: batch_norm_elemt uses invstd (inverse std), not variance
        torch::Tensor invstd;
        if (offset < Size) {
            invstd = fuzzer_utils::createTensor(Data, Size, offset);
            invstd = invstd.flatten();
            if (invstd.numel() >= num_channels) {
                invstd = invstd.slice(0, 0, num_channels).to(dtype);
            } else {
                invstd = torch::ones({num_channels}, dtype);
            }
            // Ensure invstd is positive (it's inverse std dev)
            invstd = torch::abs(invstd) + 1e-5f;
        } else {
            invstd = torch::ones({num_channels}, dtype);
        }
        
        // Get epsilon value from the input data
        float eps = 1e-5f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure epsilon is positive and reasonable
            eps = std::abs(eps);
            if (eps > 1.0f) eps = 1.0f;
            if (eps < 1e-10f) eps = 1e-10f;
        }
        
        // Apply batch_norm_elemt
        // Signature: batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        torch::Tensor output = torch::batch_norm_elemt(input, weight, bias, mean, invstd, eps);
        
        // Access the output to ensure computation is not optimized away
        volatile float sum = output.sum().item<float>();
        (void)sum;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}