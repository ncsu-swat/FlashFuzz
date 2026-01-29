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
        
        // Need at least a few bytes for the input tensor and parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor and ensure it's float type
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        input_tensor = input_tensor.to(torch::kFloat32);
        
        // Extract parameters for fake_quantize_per_tensor_affine
        // We need: scale, zero_point, quant_min, quant_max
        
        // Parse scale (positive double)
        double scale = 0.1;  // Default value
        if (offset + sizeof(float) <= Size) {
            float scale_f;
            std::memcpy(&scale_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure scale is positive and finite
            if (!std::isfinite(scale_f) || scale_f <= 0) {
                scale = 0.1;
            } else {
                scale = static_cast<double>(scale_f);
            }
            // Avoid extremely small or large values
            if (scale < 1e-10) scale = 1e-10;
            if (scale > 1e10) scale = 1e10;
        }
        
        // Parse zero_point (int64_t)
        int64_t zero_point = 0;  // Default value
        if (offset + sizeof(int32_t) <= Size) {
            int32_t zp;
            std::memcpy(&zp, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            zero_point = static_cast<int64_t>(zp);
        }
        
        // Parse quant_min and quant_max (typically 8-bit quantization range)
        int64_t quant_min = 0;
        int64_t quant_max = 255;
        if (offset + 2 <= Size) {
            // Use bytes to control quantization range type
            uint8_t range_type = Data[offset++];
            if (range_type % 3 == 0) {
                // Unsigned 8-bit: [0, 255]
                quant_min = 0;
                quant_max = 255;
            } else if (range_type % 3 == 1) {
                // Signed 8-bit: [-128, 127]
                quant_min = -128;
                quant_max = 127;
            } else {
                // Custom range from fuzzer data
                if (offset + 2 <= Size) {
                    int8_t min_val, max_val;
                    std::memcpy(&min_val, Data + offset, sizeof(int8_t));
                    offset += sizeof(int8_t);
                    std::memcpy(&max_val, Data + offset, sizeof(int8_t));
                    offset += sizeof(int8_t);
                    quant_min = static_cast<int64_t>(min_val);
                    quant_max = static_cast<int64_t>(max_val);
                }
            }
        }
        
        // Ensure quant_min < quant_max (required by the API)
        if (quant_min >= quant_max) {
            quant_min = 0;
            quant_max = 255;
        }
        
        // Ensure zero_point is within [quant_min, quant_max]
        if (zero_point < quant_min) zero_point = quant_min;
        if (zero_point > quant_max) zero_point = quant_max;
        
        // Apply fake_quantize_per_tensor_affine
        torch::Tensor output = torch::fake_quantize_per_tensor_affine(
            input_tensor, scale, zero_point, quant_min, quant_max);
        
        // Access output to ensure computation is performed
        (void)output.sum().item<float>();
        
        // Try with different tensor shapes based on fuzzer input
        if (offset + 1 <= Size) {
            uint8_t tensor_variant = Data[offset++];
            
            try {
                if (tensor_variant % 4 == 0) {
                    // Scalar tensor
                    torch::Tensor scalar_input = torch::tensor(3.14f);
                    torch::Tensor scalar_output = torch::fake_quantize_per_tensor_affine(
                        scalar_input, scale, zero_point, quant_min, quant_max);
                    (void)scalar_output.item<float>();
                } else if (tensor_variant % 4 == 1) {
                    // 1D tensor
                    torch::Tensor tensor_1d = torch::rand({5});
                    torch::Tensor output_1d = torch::fake_quantize_per_tensor_affine(
                        tensor_1d, scale, zero_point, quant_min, quant_max);
                    (void)output_1d.sum().item<float>();
                } else if (tensor_variant % 4 == 2) {
                    // 2D tensor (typical for weights)
                    torch::Tensor tensor_2d = torch::rand({3, 4});
                    torch::Tensor output_2d = torch::fake_quantize_per_tensor_affine(
                        tensor_2d, scale, zero_point, quant_min, quant_max);
                    (void)output_2d.sum().item<float>();
                } else {
                    // 4D tensor (typical for conv weights)
                    torch::Tensor tensor_4d = torch::rand({2, 3, 4, 4});
                    torch::Tensor output_4d = torch::fake_quantize_per_tensor_affine(
                        tensor_4d, scale, zero_point, quant_min, quant_max);
                    (void)output_4d.sum().item<float>();
                }
            } catch (...) {
                // Silently catch shape/type related errors
            }
        }
        
        // Test with different scale values
        if (offset + 1 <= Size) {
            uint8_t scale_variant = Data[offset++];
            double test_scale = scale;
            
            if (scale_variant % 4 == 0) {
                test_scale = 1e-5;  // Small scale
            } else if (scale_variant % 4 == 1) {
                test_scale = 1.0;   // Unit scale
            } else if (scale_variant % 4 == 2) {
                test_scale = 100.0; // Large scale
            }
            // else keep original scale
            
            try {
                torch::Tensor test_output = torch::fake_quantize_per_tensor_affine(
                    input_tensor, test_scale, zero_point, quant_min, quant_max);
                (void)test_output.sum().item<float>();
            } catch (...) {
                // Silently catch errors from edge cases
            }
        }
        
        // Test with tensors containing special values
        if (offset + 1 <= Size) {
            uint8_t special_variant = Data[offset++];
            
            try {
                torch::Tensor special_tensor;
                if (special_variant % 4 == 0) {
                    // Tensor with zeros
                    special_tensor = torch::zeros({3, 4});
                } else if (special_variant % 4 == 1) {
                    // Tensor with ones
                    special_tensor = torch::ones({3, 4});
                } else if (special_variant % 4 == 2) {
                    // Tensor with negative values
                    special_tensor = torch::randn({3, 4});
                } else {
                    // Tensor with large values
                    special_tensor = torch::rand({3, 4}) * 1000.0f;
                }
                
                torch::Tensor special_output = torch::fake_quantize_per_tensor_affine(
                    special_tensor, scale, zero_point, quant_min, quant_max);
                (void)special_output.sum().item<float>();
            } catch (...) {
                // Silently catch errors
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}