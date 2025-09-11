#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input tensor has at least 5 dimensions for FractionalMaxPool3d
        // If not, reshape it to have 5 dimensions (batch, channels, d, h, w)
        if (input.dim() < 5) {
            std::vector<int64_t> new_shape(5, 1);
            int64_t total_elements = input.numel();
            
            // Try to distribute elements across the 5D shape
            if (total_elements > 0) {
                new_shape[0] = 1; // batch
                new_shape[1] = 1; // channels
                
                // Distribute remaining elements across spatial dimensions
                int64_t spatial_elements = total_elements;
                new_shape[2] = std::max(static_cast<int64_t>(1), static_cast<int64_t>(std::cbrt(spatial_elements)));
                spatial_elements /= new_shape[2];
                new_shape[3] = std::max(static_cast<int64_t>(1), static_cast<int64_t>(std::sqrt(spatial_elements)));
                new_shape[4] = std::max(static_cast<int64_t>(1), spatial_elements / new_shape[3]);
            }
            
            // Reshape the tensor
            input = input.reshape(new_shape);
        }
        
        // Extract parameters for FractionalMaxPool3d from the input data
        double kernel_size_d = 2.0;
        double kernel_size_h = 2.0;
        double kernel_size_w = 2.0;
        double output_ratio_d = 0.5;
        double output_ratio_h = 0.5;
        double output_ratio_w = 0.5;
        
        // Parse parameters if we have enough data
        if (offset + 24 <= Size) {
            // Extract kernel sizes
            std::memcpy(&kernel_size_d, Data + offset, sizeof(double));
            offset += sizeof(double);
            std::memcpy(&kernel_size_h, Data + offset, sizeof(double));
            offset += sizeof(double);
            std::memcpy(&kernel_size_w, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure kernel sizes are positive
            kernel_size_d = std::abs(kernel_size_d);
            kernel_size_h = std::abs(kernel_size_h);
            kernel_size_w = std::abs(kernel_size_w);
            
            // Limit to reasonable range
            kernel_size_d = std::fmod(kernel_size_d, 5.0) + 1.0;
            kernel_size_h = std::fmod(kernel_size_h, 5.0) + 1.0;
            kernel_size_w = std::fmod(kernel_size_w, 5.0) + 1.0;
            
            // Extract output ratios
            std::memcpy(&output_ratio_d, Data + offset, sizeof(double));
            offset += sizeof(double);
            std::memcpy(&output_ratio_h, Data + offset, sizeof(double));
            offset += sizeof(double);
            std::memcpy(&output_ratio_w, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure output ratios are between 0 and 1
            output_ratio_d = std::abs(output_ratio_d);
            output_ratio_h = std::abs(output_ratio_h);
            output_ratio_w = std::abs(output_ratio_w);
            
            output_ratio_d = std::fmod(output_ratio_d, 1.0);
            output_ratio_h = std::fmod(output_ratio_h, 1.0);
            output_ratio_w = std::fmod(output_ratio_w, 1.0);
            
            // Ensure output ratios are not too small
            output_ratio_d = std::max(0.1, output_ratio_d);
            output_ratio_h = std::max(0.1, output_ratio_h);
            output_ratio_w = std::max(0.1, output_ratio_w);
        }
        
        // Create FractionalMaxPool3d options
        torch::nn::FractionalMaxPool3dOptions options(
            {kernel_size_d, kernel_size_h, kernel_size_w}
        );
        options.output_ratio({output_ratio_d, output_ratio_h, output_ratio_w});
        
        // Create the FractionalMaxPool3d module
        torch::nn::FractionalMaxPool3d pool(options);
        
        // Apply the operation
        auto output = pool->forward(input);
        
        // Use the output to ensure it's not optimized away
        auto sum = output.sum();
        if (sum.item<float>() == -1.0f) {
            // This will never happen, just to prevent compiler optimization
            throw std::runtime_error("Unexpected sum value");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
