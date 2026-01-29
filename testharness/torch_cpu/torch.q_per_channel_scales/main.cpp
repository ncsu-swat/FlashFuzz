#include "fuzzer_utils.h"
#include <iostream>

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

        // Need enough data for meaningful fuzzing
        if (Size < 8) {
            return 0;
        }

        // Read parameters from fuzzer data
        uint8_t num_channels = (Data[offset++] % 8) + 1;  // 1-8 channels
        uint8_t axis_byte = Data[offset++];
        uint8_t dim_config = Data[offset++];

        // Determine tensor dimensions (2D, 3D, or 4D)
        int num_dims = (dim_config % 3) + 2;  // 2, 3, or 4 dimensions
        
        // Build shape with num_channels at the specified axis
        std::vector<int64_t> shape;
        int64_t axis = axis_byte % num_dims;
        
        for (int i = 0; i < num_dims; i++) {
            if (i == axis) {
                shape.push_back(num_channels);
            } else {
                // Small dimensions: 2-4
                shape.push_back((offset < Size ? (Data[offset++] % 3) : 0) + 2);
            }
        }

        // Create the input tensor to be quantized (float type required)
        torch::Tensor input_tensor = torch::rand(shape, torch::kFloat32);

        // Create scales tensor (must be 1D with size = num_channels, positive values)
        std::vector<double> scales_data;
        for (int i = 0; i < num_channels; i++) {
            // Generate positive scale values from fuzzer data
            double scale_val = 0.01;
            if (offset < Size) {
                scale_val = (static_cast<double>(Data[offset++]) / 255.0) * 0.99 + 0.01;  // Range [0.01, 1.0]
            }
            scales_data.push_back(scale_val);
        }
        torch::Tensor scales = torch::tensor(scales_data, torch::kFloat64);

        // Create zero_points tensor (must be 1D with size = num_channels, integer values)
        std::vector<int64_t> zp_data;
        for (int i = 0; i < num_channels; i++) {
            int64_t zp_val = 0;
            if (offset < Size) {
                zp_val = static_cast<int64_t>(Data[offset++]) - 128;  // Range [-128, 127]
            }
            zp_data.push_back(zp_val);
        }
        torch::Tensor zero_points = torch::tensor(zp_data, torch::kInt64);

        // Create a per-channel quantized tensor
        torch::Tensor quantized_tensor;
        try {
            quantized_tensor = torch::quantize_per_channel(
                input_tensor,
                scales,
                zero_points,
                axis,
                torch::kQInt8  // or kQUInt8
            );
        } catch (const c10::Error& e) {
            // Shape mismatches or invalid configurations are expected
            return 0;
        } catch (const std::exception& e) {
            // Other expected failures during quantization
            return 0;
        }

        // Now test q_per_channel_scales on the quantized tensor
        torch::Tensor retrieved_scales = torch::q_per_channel_scales(quantized_tensor);

        // Verify the result has expected properties
        if (retrieved_scales.dim() != 1) {
            std::cerr << "Unexpected: scales should be 1D" << std::endl;
        }
        if (retrieved_scales.size(0) != num_channels) {
            std::cerr << "Unexpected: scales size mismatch" << std::endl;
        }

        // Also test q_per_channel_zero_points and q_per_channel_axis for coverage
        torch::Tensor retrieved_zp = torch::q_per_channel_zero_points(quantized_tensor);
        int64_t retrieved_axis = torch::q_per_channel_axis(quantized_tensor);

        // Basic sanity checks
        (void)retrieved_zp.numel();
        (void)retrieved_axis;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}