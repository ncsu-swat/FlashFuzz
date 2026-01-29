#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor and convert to float (required for quantization)
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        input_tensor = input_tensor.to(torch::kFloat32).contiguous();
        
        // Extract quantization parameters from the remaining data
        uint8_t scale_byte = 0;
        uint8_t zero_point_byte = 0;
        uint8_t dtype_byte = 0;
        
        if (offset + 3 <= Size) {
            scale_byte = Data[offset++];
            zero_point_byte = Data[offset++];
            dtype_byte = Data[offset++];
        }
        
        // Generate scale value (between 1e-5 and 1.0, must be positive)
        double scale = 1e-5 + (scale_byte % 100) * 0.01;
        
        // Generate zero_point value (between -128 and 127 for int8, 0 and 255 for uint8)
        int64_t zero_point = 0;
        
        // Choose quantization dtype based on input
        torch::ScalarType q_dtype;
        if (dtype_byte % 3 == 0) {
            q_dtype = torch::kQInt8;
            zero_point = static_cast<int64_t>(zero_point_byte) - 128; // Range: -128 to 127
        } else if (dtype_byte % 3 == 1) {
            q_dtype = torch::kQUInt8;
            zero_point = static_cast<int64_t>(zero_point_byte); // Range: 0 to 255
        } else {
            q_dtype = torch::kQInt32;
            zero_point = static_cast<int64_t>(zero_point_byte) - 128; // Range: -128 to 127
        }
        
        // Try per tensor quantization
        try {
            torch::Tensor quantized_tensor = torch::quantize_per_tensor(
                input_tensor, scale, zero_point, q_dtype);
            
            // Dequantize to verify round trip
            torch::Tensor dequantized = torch::dequantize(quantized_tensor);
            
            // Access quantization parameters
            double q_scale = quantized_tensor.q_scale();
            int64_t q_zero_point = quantized_tensor.q_zero_point();
            (void)q_scale;
            (void)q_zero_point;
        } catch (const std::exception& e) {
            // Quantization might fail for some tensor types, that's expected
        }
        
        // Try per-channel quantization if tensor has enough dimensions
        if (input_tensor.dim() > 0 && input_tensor.numel() > 0) {
            try {
                int64_t channel_dim = 0;
                int64_t num_channels = input_tensor.size(channel_dim);
                
                if (num_channels > 0) {
                    std::vector<double> scales_vec(num_channels, scale);
                    std::vector<int64_t> zero_points_vec(num_channels, zero_point);
                    
                    torch::Tensor scales = torch::tensor(scales_vec, torch::kFloat64);
                    torch::Tensor zero_points = torch::tensor(zero_points_vec, torch::kLong);
                    
                    // Per channel quantization (only works with qint8 and quint8)
                    torch::ScalarType pc_dtype = (dtype_byte % 2 == 0) ? torch::kQInt8 : torch::kQUInt8;
                    
                    // Adjust zero_points for the chosen dtype
                    if (pc_dtype == torch::kQUInt8) {
                        for (auto& zp : zero_points_vec) {
                            zp = std::max(0L, std::min(255L, zp + 128));
                        }
                        zero_points = torch::tensor(zero_points_vec, torch::kLong);
                    }
                    
                    torch::Tensor quantized_per_channel = torch::quantize_per_channel(
                        input_tensor, scales, zero_points, channel_dim, pc_dtype);
                    
                    // Dequantize to verify round trip
                    torch::Tensor dequantized_per_channel = torch::dequantize(quantized_per_channel);
                }
            } catch (const std::exception& e) {
                // Per-channel quantization might fail for some tensor types, that's expected
            }
        }
        
        // Try fake quantization with different quant_min/quant_max ranges
        try {
            // For symmetric quantization around zero
            torch::Tensor fake_quantized = torch::fake_quantize_per_tensor_affine(
                input_tensor, scale, 0, -128, 127);
        } catch (const std::exception& e) {
            // Fake quantization might fail for some tensor types, that's expected
        }
        
        try {
            // For asymmetric quantization (unsigned)
            torch::Tensor fake_quantized_unsigned = torch::fake_quantize_per_tensor_affine(
                input_tensor, scale, 128, 0, 255);
        } catch (const std::exception& e) {
            // Fake quantization might fail for some tensor types, that's expected
        }
        
        // Try fake quantize per channel if tensor has enough dimensions
        if (input_tensor.dim() >= 2) {
            try {
                int64_t axis = 0;
                int64_t num_channels = input_tensor.size(axis);
                
                if (num_channels > 0) {
                    torch::Tensor scales = torch::full({num_channels}, scale, torch::kFloat32);
                    torch::Tensor zero_points = torch::full({num_channels}, zero_point, torch::kLong);
                    
                    // Clamp zero_points to valid range
                    zero_points = zero_points.clamp(-128, 127);
                    
                    torch::Tensor fake_quant_per_channel = torch::fake_quantize_per_channel_affine(
                        input_tensor, scales, zero_points, axis, -128, 127);
                }
            } catch (const std::exception& e) {
                // Per-channel fake quantization might fail, that's expected
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