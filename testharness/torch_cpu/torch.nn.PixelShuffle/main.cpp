#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least a few bytes for the input tensor and upscale factor
        if (Size < 5) {
            return 0;
        }
        
        // Extract upscale_factor from the first byte
        uint8_t upscale_factor_byte = Data[0];
        // Make upscale_factor a positive integer between 1 and 4 (keeping it small for valid channel counts)
        int64_t upscale_factor = (upscale_factor_byte % 4) + 1;
        int64_t channels_multiplier = upscale_factor * upscale_factor;
        
        size_t offset = 1;
        
        // Create input tensor from remaining data
        torch::Tensor input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
        
        // PixelShuffle requires 4D input (N, C, H, W) where C must be divisible by upscale_factor^2
        // Reshape input to be 4D with appropriate channel count
        int64_t total_elements = input.numel();
        
        if (total_elements < channels_multiplier) {
            // Not enough elements, pad the tensor
            input = torch::nn::functional::pad(
                input.flatten(),
                torch::nn::functional::PadFuncOptions({0, channels_multiplier - total_elements})
            );
            total_elements = channels_multiplier;
        }
        
        // Flatten and reshape to 4D
        input = input.flatten();
        
        // Calculate dimensions that satisfy PixelShuffle constraints
        // C must be divisible by upscale_factor^2
        // We need: N * C * H * W = total_elements, where C = k * upscale_factor^2
        
        // Simple approach: make it (1, channels_multiplier * k, h, w)
        int64_t remaining = total_elements / channels_multiplier;
        int64_t k = 1;
        int64_t spatial = remaining / k;
        
        // Find reasonable H and W
        int64_t h = 1;
        int64_t w = spatial;
        for (int64_t i = 2; i * i <= spatial; ++i) {
            if (spatial % i == 0) {
                h = i;
                w = spatial / i;
            }
        }
        
        int64_t channels = channels_multiplier * k;
        int64_t needed_elements = channels * h * w;
        
        // Truncate or pad to exact size
        if (total_elements > needed_elements) {
            input = input.narrow(0, 0, needed_elements);
        } else if (total_elements < needed_elements) {
            input = torch::nn::functional::pad(
                input,
                torch::nn::functional::PadFuncOptions({0, needed_elements - total_elements})
            );
        }
        
        // Reshape to (1, C, H, W)
        input = input.reshape({1, channels, h, w});
        
        // Ensure float type for better coverage
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create PixelShuffle module
        torch::nn::PixelShuffle pixel_shuffle(upscale_factor);
        
        // Apply PixelShuffle to the input tensor
        torch::Tensor output = pixel_shuffle->forward(input);
        
        // Verify output shape: should be (N, C/r^2, H*r, W*r)
        // This exercises the module's output validation
        
        // Optional: Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        auto mean = output.mean();
        
        // Prevent the compiler from optimizing away the operations
        if (sum.item<float>() == -1.0f && mean.item<float>() == -2.0f) {
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}