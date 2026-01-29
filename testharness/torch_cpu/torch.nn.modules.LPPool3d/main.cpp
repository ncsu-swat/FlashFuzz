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
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters for LPPool3d from the data first
        uint8_t norm_type_byte = Data[offset++];
        double norm_type = static_cast<double>(norm_type_byte % 10) + 1.0; // Norm type between 1 and 10
        
        // Extract kernel size
        int64_t kernel_d = static_cast<int64_t>(Data[offset++] % 5) + 1; // 1-5
        int64_t kernel_h = static_cast<int64_t>(Data[offset++] % 5) + 1; // 1-5
        int64_t kernel_w = static_cast<int64_t>(Data[offset++] % 5) + 1; // 1-5
        
        // Extract stride
        int64_t stride_d = static_cast<int64_t>(Data[offset++] % 3) + 1; // 1-3
        int64_t stride_h = static_cast<int64_t>(Data[offset++] % 3) + 1; // 1-3
        int64_t stride_w = static_cast<int64_t>(Data[offset++] % 3) + 1; // 1-3
        
        // Extract ceil_mode
        bool ceil_mode = Data[offset++] % 2 == 1;
        
        // Create input tensor from remaining data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 5D tensor (batch, channels, depth, height, width)
        // Make sure dimensions are large enough for the kernel
        int64_t min_d = kernel_d + 1;
        int64_t min_h = kernel_h + 1;
        int64_t min_w = kernel_w + 1;
        
        int64_t total_elements = input.numel();
        if (total_elements < min_d * min_h * min_w) {
            // Pad with zeros if necessary
            input = torch::zeros({1, 1, min_d, min_h, min_w});
        } else {
            // Reshape to 5D with reasonable dimensions
            int64_t batch = 1;
            int64_t channels = 1;
            int64_t depth = min_d;
            int64_t height = min_h;
            int64_t width = total_elements / (batch * channels * depth * height);
            if (width < min_w) width = min_w;
            
            // Resize input to fit the required shape
            int64_t needed = batch * channels * depth * height * width;
            if (total_elements < needed) {
                input = torch::zeros({batch, channels, depth, height, width});
            } else {
                input = input.flatten().slice(0, 0, needed).reshape({batch, channels, depth, height, width});
            }
        }
        
        // Make input contiguous and float
        input = input.to(torch::kFloat32).contiguous();
        
        // Create LPPool3d modules with different configurations
        torch::nn::LPPool3d lppool_single(
            torch::nn::LPPool3dOptions(norm_type, kernel_d)
                .stride(stride_d)
                .ceil_mode(ceil_mode)
        );
        
        torch::nn::LPPool3d lppool_triple(
            torch::nn::LPPool3dOptions(norm_type, {kernel_d, kernel_h, kernel_w})
                .stride({stride_d, stride_h, stride_w})
                .ceil_mode(ceil_mode)
        );
        
        // Apply the LPPool3d operations with inner try-catch for expected shape errors
        try {
            torch::Tensor output_single = lppool_single->forward(input);
            auto sum_single = output_single.sum();
            (void)sum_single;
        } catch (const c10::Error&) {
            // Shape mismatch is expected for some input configurations
        }
        
        try {
            torch::Tensor output_triple = lppool_triple->forward(input);
            auto sum_triple = output_triple.sum();
            (void)sum_triple;
        } catch (const c10::Error&) {
            // Shape mismatch is expected for some input configurations
        }
        
        // Test with different norm types
        try {
            torch::nn::LPPool3d lppool_norm2(
                torch::nn::LPPool3dOptions(2.0, {kernel_d, kernel_h, kernel_w})
                    .stride({stride_d, stride_h, stride_w})
                    .ceil_mode(!ceil_mode)
            );
            torch::Tensor output_norm2 = lppool_norm2->forward(input);
            auto sum_norm2 = output_norm2.sum();
            (void)sum_norm2;
        } catch (const c10::Error&) {
            // Expected for some configurations
        }
        
        // Test with infinity norm
        try {
            torch::nn::LPPool3d lppool_inf(
                torch::nn::LPPool3dOptions(std::numeric_limits<double>::infinity(), {kernel_d, kernel_h, kernel_w})
                    .stride({stride_d, stride_h, stride_w})
                    .ceil_mode(ceil_mode)
            );
            torch::Tensor output_inf = lppool_inf->forward(input);
            auto sum_inf = output_inf.sum();
            (void)sum_inf;
        } catch (const c10::Error&) {
            // Expected for some configurations
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}