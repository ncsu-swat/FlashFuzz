#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Extract control bytes first
        uint8_t mode_selector = Data[offset++] % 4;
        bool use_scale_factors = Data[offset++] & 0x1;
        bool align_corners = Data[offset++] & 0x1;
        uint8_t dim_selector = Data[offset++] % 3; // 0=3D, 1=4D, 2=5D
        
        // Determine target dimensions based on mode
        int64_t target_dim;
        bool supports_align_corners = true;
        
        // Build Upsample options with mode
        torch::nn::UpsampleOptions options;
        
        switch (mode_selector) {
            case 0:
                options = options.mode(torch::kNearest);
                target_dim = 3 + dim_selector; // nearest works with 3D, 4D, 5D
                supports_align_corners = false; // align_corners not supported for nearest
                break;
            case 1:
                options = options.mode(torch::kLinear);
                target_dim = 3; // linear requires 3D (batch, channel, length)
                break;
            case 2:
                options = options.mode(torch::kBilinear);
                target_dim = 4; // bilinear requires 4D (batch, channel, height, width)
                break;
            case 3:
                options = options.mode(torch::kTrilinear);
                target_dim = 5; // trilinear requires 5D (batch, channel, depth, height, width)
                break;
            default:
                options = options.mode(torch::kNearest);
                target_dim = 3;
                supports_align_corners = false;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape to target dimensions
        auto input_sizes = input_tensor.sizes();
        int64_t total_elements = input_tensor.numel();
        
        if (total_elements < 1) {
            return 0;
        }
        
        // Build shape for target_dim dimensions
        std::vector<int64_t> new_shape;
        new_shape.push_back(1); // batch size
        new_shape.push_back(1); // channels
        
        // Distribute remaining elements across spatial dimensions
        int64_t spatial_dims = target_dim - 2;
        int64_t spatial_size = std::max(int64_t(1), (int64_t)std::pow(total_elements, 1.0 / spatial_dims));
        
        for (int i = 0; i < spatial_dims - 1; i++) {
            new_shape.push_back(spatial_size);
        }
        // Last spatial dim gets whatever is left
        int64_t remaining = total_elements;
        for (int i = 0; i < spatial_dims - 1; i++) {
            remaining /= spatial_size;
        }
        new_shape.push_back(std::max(int64_t(1), remaining));
        
        // Compute actual total and adjust
        int64_t computed_total = 1;
        for (auto s : new_shape) computed_total *= s;
        
        // Flatten and take needed elements, then reshape
        input_tensor = input_tensor.flatten().narrow(0, 0, std::min(total_elements, computed_total));
        if (input_tensor.numel() < computed_total) {
            // Pad with zeros if needed
            auto padding = torch::zeros({computed_total - input_tensor.numel()}, input_tensor.options());
            input_tensor = torch::cat({input_tensor, padding});
        }
        input_tensor = input_tensor.reshape(new_shape);
        input_sizes = input_tensor.sizes();
        
        // align_corners only valid for linear, bilinear, bicubic, trilinear
        if (supports_align_corners) {
            options = options.align_corners(align_corners);
        }
        
        if (use_scale_factors) {
            // Use scale factors
            std::vector<double> scale_factors;
            for (int64_t i = 2; i < target_dim; i++) {
                double scale = 1.5; // default
                if (offset + sizeof(uint8_t) <= Size) {
                    // Map byte to reasonable scale factor range [0.5, 4.0]
                    uint8_t scale_byte = Data[offset++];
                    scale = 0.5 + (scale_byte / 255.0) * 3.5;
                }
                scale_factors.push_back(scale);
            }
            options = options.scale_factor(scale_factors);
        } else {
            // Use output size
            std::vector<int64_t> output_size;
            for (int64_t i = 2; i < target_dim; i++) {
                int64_t size = 4; // default
                if (offset + sizeof(uint8_t) <= Size) {
                    // Map byte to reasonable output size range [1, 64]
                    size = 1 + (Data[offset++] % 64);
                }
                output_size.push_back(size);
            }
            options = options.size(output_size);
        }
        
        // Create and apply the Upsample module
        torch::nn::Upsample upsample(options);
        
        try {
            torch::Tensor output = upsample->forward(input_tensor);
            
            // Ensure the output is used to prevent optimization
            auto sum = output.sum().item<float>();
            (void)sum;
        } catch (const c10::Error&) {
            // Inner catch for expected PyTorch errors (shape issues, etc.)
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}