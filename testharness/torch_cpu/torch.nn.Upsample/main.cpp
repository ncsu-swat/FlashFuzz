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
        
        if (Size < 10) return 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is floating point for interpolation
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Get input dimensions - interpolate requires 3D, 4D, or 5D
        int64_t dim = input.dim();
        
        // Reshape to valid dimensions if needed (3D, 4D, or 5D)
        if (dim < 3) {
            // Reshape to 3D: (batch, channels, length)
            int64_t total = input.numel();
            if (total == 0) return 0;
            input = input.reshape({1, 1, total});
            dim = 3;
        } else if (dim > 5) {
            // Flatten extra dims into spatial dimension
            auto sizes = input.sizes();
            int64_t batch = sizes[0];
            int64_t channels = sizes[1];
            int64_t spatial = 1;
            for (int i = 2; i < dim; i++) {
                spatial *= sizes[i];
            }
            input = input.reshape({batch, channels, spatial});
            dim = 3;
        }
        
        // Parse mode selector
        if (offset >= Size) return 0;
        uint8_t mode_selector = Data[offset++];
        
        // Select mode based on dimensions
        // Track whether mode supports align_corners
        bool supports_align_corners = false;
        torch::nn::functional::InterpolateFuncOptions::mode_t mode;
        if (dim == 3) {
            // 3D: nearest or linear
            if (mode_selector % 2 == 0) {
                mode = torch::kNearest;
                supports_align_corners = false;
            } else {
                mode = torch::kLinear;
                supports_align_corners = true;
            }
        } else if (dim == 4) {
            // 4D: nearest, bilinear, or bicubic
            int choice = mode_selector % 3;
            if (choice == 0) {
                mode = torch::kNearest;
                supports_align_corners = false;
            } else if (choice == 1) {
                mode = torch::kBilinear;
                supports_align_corners = true;
            } else {
                mode = torch::kBicubic;
                supports_align_corners = true;
            }
        } else {
            // 5D: nearest or trilinear
            if (mode_selector % 2 == 0) {
                mode = torch::kNearest;
                supports_align_corners = false;
            } else {
                mode = torch::kTrilinear;
                supports_align_corners = true;
            }
        }
        
        // Parse align_corners flag
        bool align_corners = false;
        if (offset < Size) {
            align_corners = Data[offset++] & 1;
        }
        
        // Parse whether to use scale_factor or size
        bool use_scale_factor = false;
        if (offset < Size) {
            use_scale_factor = Data[offset++] & 1;
        }
        
        // Create interpolate options
        torch::nn::functional::InterpolateFuncOptions options;
        options.mode(mode);
        
        // align_corners only valid for linear, bilinear, bicubic, trilinear
        if (supports_align_corners) {
            options.align_corners(align_corners);
        }
        
        int spatial_dims = dim - 2;
        
        if (use_scale_factor) {
            std::vector<double> scale_factors;
            for (int i = 0; i < spatial_dims; i++) {
                double scale = 2.0; // default
                if (offset < Size) {
                    // Generate scale factor between 0.5 and 4.0
                    uint8_t scale_byte = Data[offset++];
                    scale = 0.5 + (scale_byte / 255.0) * 3.5;
                }
                scale_factors.push_back(scale);
            }
            options.scale_factor(scale_factors);
        } else {
            std::vector<int64_t> output_size;
            auto input_sizes = input.sizes();
            for (int i = 0; i < spatial_dims; i++) {
                int64_t size_val = 4; // default
                if (offset < Size) {
                    uint8_t size_byte = Data[offset++];
                    // Generate size between 1 and 64
                    size_val = (size_byte % 64) + 1;
                }
                output_size.push_back(size_val);
            }
            options.size(output_size);
        }
        
        // Apply interpolate operation
        torch::Tensor output;
        try {
            output = torch::nn::functional::interpolate(input, options);
        } catch (const c10::Error& e) {
            // Expected errors for invalid configurations
            return 0;
        }
        
        // Verify output and perform operations
        if (output.defined() && output.numel() > 0) {
            auto sum = output.sum();
            auto mean = output.mean();
            
            // Test backward pass
            torch::Tensor input_grad = input.detach().requires_grad_(true);
            try {
                auto out_grad = torch::nn::functional::interpolate(input_grad, options);
                out_grad.sum().backward();
            } catch (...) {
                // Gradient computation may fail for some modes
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}