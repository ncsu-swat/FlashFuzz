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
        
        // Need at least a few bytes for basic operations
        if (Size < 8) {
            return 0;
        }
        
        // Parse configuration bytes first
        uint8_t config_byte = Data[offset++];
        uint8_t padding_left_raw = Data[offset++];
        uint8_t padding_right_raw = Data[offset++];
        uint8_t dim_config = Data[offset++];
        
        // Create input tensor with appropriate dimensions for CircularPad1d
        // CircularPad1d expects 2D (unbatched: C x W) or 3D (batched: N x C x W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape input to be 2D or 3D as required by CircularPad1d
        int64_t total_elements = input.numel();
        if (total_elements < 2) {
            return 0;  // Need at least some elements
        }
        
        // Decide on 2D or 3D based on config
        bool use_3d = (dim_config & 0x1);
        
        if (use_3d) {
            // 3D: N x C x W - try to make reasonable batch/channel/width
            int64_t batch_size = std::max(static_cast<int64_t>(1), static_cast<int64_t>((dim_config >> 1) % 4 + 1));
            int64_t remaining = total_elements / batch_size;
            int64_t channels = std::max(static_cast<int64_t>(1), remaining / 4);
            int64_t width = std::max(static_cast<int64_t>(1), remaining / channels);
            
            // Ensure we have valid dimensions
            int64_t needed = batch_size * channels * width;
            if (needed > total_elements) {
                width = total_elements / (batch_size * channels);
                if (width < 1) width = 1;
            }
            needed = batch_size * channels * width;
            
            input = input.flatten().narrow(0, 0, needed).view({batch_size, channels, width});
        } else {
            // 2D: C x W
            int64_t channels = std::max(static_cast<int64_t>(1), static_cast<int64_t>((dim_config >> 1) % 8 + 1));
            int64_t width = total_elements / channels;
            if (width < 1) width = 1;
            int64_t needed = channels * width;
            
            input = input.flatten().narrow(0, 0, needed).view({channels, width});
        }
        
        // Get the last dimension size (width) for padding validation
        int64_t input_width = input.size(-1);
        
        // Limit padding to be reasonable relative to input width
        // Circular padding requires padding <= input_width
        int64_t max_padding = std::max(static_cast<int64_t>(1), input_width);
        int64_t padding_left = static_cast<int64_t>(padding_left_raw) % max_padding;
        int64_t padding_right = static_cast<int64_t>(padding_right_raw) % max_padding;
        
        // For 1D padding, we need a 2-element vector: {left, right}
        // Note: In PyTorch functional::pad, the padding is specified from last dim to first
        // For 1D: {pad_left, pad_right} for the last dimension
        
        // Test with different padding configurations
        if (config_byte & 0x1) {
            // Symmetric padding (same on both sides)
            padding_right = padding_left;
        }
        
        // Apply circular padding using the functional interface
        // torch::nn::functional::pad is the C++ equivalent of F.pad in Python
        torch::Tensor output = torch::nn::functional::pad(
            input, 
            torch::nn::functional::PadFuncOptions({padding_left, padding_right}).mode(torch::kCircular)
        );
        
        // Ensure the output is used to prevent optimization
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        // Test with asymmetric padding for better coverage
        try {
            int64_t asym_left = (padding_left + 1) % max_padding;
            int64_t asym_right = (padding_right + 2) % max_padding;
            torch::Tensor output_asym = torch::nn::functional::pad(
                input, 
                torch::nn::functional::PadFuncOptions({asym_left, asym_right}).mode(torch::kCircular)
            );
            if (output_asym.numel() > 0) {
                volatile float sum2 = output_asym.sum().item<float>();
                (void)sum2;
            }
        } catch (...) {
            // Silently ignore failures in variant testing
        }
        
        // Test with zero padding on one side
        try {
            torch::Tensor output_zero_left = torch::nn::functional::pad(
                input, 
                torch::nn::functional::PadFuncOptions({0, padding_right}).mode(torch::kCircular)
            );
            if (output_zero_left.numel() > 0) {
                volatile float sum3 = output_zero_left.sum().item<float>();
                (void)sum3;
            }
        } catch (...) {
            // Silently ignore failures in variant testing
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}