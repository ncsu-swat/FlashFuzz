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
        
        // Need at least a few bytes for the input tensor and parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for fake_quantize_per_tensor_affine
        // We need: scale, zero_point, quant_min, quant_max
        
        // Parse scale (positive float)
        float scale = 0.1f;  // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure scale is positive
            scale = std::abs(scale);
            // Avoid extremely small or large values
            if (scale < 1e-10f) scale = 1e-10f;
            if (scale > 1e10f) scale = 1e10f;
        }
        
        // Parse zero_point (int)
        int64_t zero_point = 0;  // Default value
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse quant_min (int)
        int64_t quant_min = 0;  // Default value
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&quant_min, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Typical range for quant_min is 0 or -128
        }
        
        // Parse quant_max (int)
        int64_t quant_max = 255;  // Default value
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&quant_max, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Typical range for quant_max is 255 or 127
        }
        
        // Ensure quant_min < quant_max
        if (quant_min >= quant_max) {
            std::swap(quant_min, quant_max);
        }
        
        // Apply fake_quantize_per_tensor_affine
        torch::Tensor output = torch::fake_quantize_per_tensor_affine(
            input_tensor, scale, zero_point, quant_min, quant_max);
        
        // Try to access the output tensor to ensure computation is performed
        auto output_accessor = output.accessor<float, 1>();
        
        // Try with different tensor types
        if (offset + 1 <= Size) {
            uint8_t tensor_variant = Data[offset++];
            
            // Create a different tensor type based on the variant
            if (tensor_variant % 3 == 0) {
                // Try with a scalar tensor
                torch::Tensor scalar_input = torch::tensor(3.14f);
                torch::Tensor scalar_output = torch::fake_quantize_per_tensor_affine(
                    scalar_input, scale, zero_point, quant_min, quant_max);
            } else if (tensor_variant % 3 == 1) {
                // Try with a 1D tensor
                torch::Tensor tensor_1d = torch::rand({5});
                torch::Tensor output_1d = torch::fake_quantize_per_tensor_affine(
                    tensor_1d, scale, zero_point, quant_min, quant_max);
            } else {
                // Try with a multi-dimensional tensor
                torch::Tensor tensor_nd = torch::rand({2, 3, 4});
                torch::Tensor output_nd = torch::fake_quantize_per_tensor_affine(
                    tensor_nd, scale, zero_point, quant_min, quant_max);
            }
        }
        
        // Try edge cases with extreme values
        if (offset + 1 <= Size) {
            uint8_t edge_case = Data[offset++];
            
            if (edge_case % 5 == 0) {
                // Very small scale
                torch::Tensor output_small_scale = torch::fake_quantize_per_tensor_affine(
                    input_tensor, 1e-10f, zero_point, quant_min, quant_max);
            } else if (edge_case % 5 == 1) {
                // Very large scale
                torch::Tensor output_large_scale = torch::fake_quantize_per_tensor_affine(
                    input_tensor, 1e10f, zero_point, quant_min, quant_max);
            } else if (edge_case % 5 == 2) {
                // Extreme zero_point
                torch::Tensor output_extreme_zp = torch::fake_quantize_per_tensor_affine(
                    input_tensor, scale, std::numeric_limits<int64_t>::max() / 2, quant_min, quant_max);
            } else if (edge_case % 5 == 3) {
                // Equal quant_min and quant_max
                torch::Tensor output_equal_range = torch::fake_quantize_per_tensor_affine(
                    input_tensor, scale, zero_point, 10, 10);
            } else {
                // Inverted quant_min and quant_max
                torch::Tensor output_inverted_range = torch::fake_quantize_per_tensor_affine(
                    input_tensor, scale, zero_point, quant_max, quant_min);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
