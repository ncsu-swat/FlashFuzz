#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract quantization parameters from the remaining data
        uint8_t scale_byte = 0;
        uint8_t zero_point_byte = 0;
        uint8_t dtype_byte = 0;
        
        if (offset + 3 <= Size) {
            scale_byte = Data[offset++];
            zero_point_byte = Data[offset++];
            dtype_byte = Data[offset++];
        }
        
        // Generate scale value (between 1e-5 and 1.0)
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
        
        // Try different quantization methods
        try {
            // Per tensor quantization
            torch::Tensor quantized_tensor = torch::quantize_per_tensor(
                input_tensor, scale, zero_point, q_dtype);
            
            // Dequantize to verify round trip
            torch::Tensor dequantized = torch::dequantize(quantized_tensor);
        } catch (const std::exception& e) {
            // Quantization might fail for some tensor types, that's expected
        }
        
        // Try per-channel quantization if tensor has enough dimensions
        if (input_tensor.dim() > 0) {
            try {
                // Create scales and zero_points tensors for per-channel quantization
                int64_t channel_dim = input_tensor.dim() > 0 ? 0 : 0;
                int64_t num_channels = input_tensor.dim() > 0 ? input_tensor.size(channel_dim) : 1;
                
                if (num_channels > 0) {
                    std::vector<double> scales_vec(num_channels, scale);
                    std::vector<int64_t> zero_points_vec(num_channels, zero_point);
                    
                    torch::Tensor scales = torch::from_blob(
                        scales_vec.data(), {num_channels}, torch::kFloat64).clone();
                    torch::Tensor zero_points = torch::from_blob(
                        zero_points_vec.data(), {num_channels}, torch::kLong).clone();
                    
                    // Per channel quantization
                    torch::Tensor quantized_per_channel = torch::quantize_per_channel(
                        input_tensor, scales, zero_points, channel_dim, q_dtype);
                    
                    // Dequantize to verify round trip
                    torch::Tensor dequantized_per_channel = torch::dequantize(quantized_per_channel);
                }
            } catch (const std::exception& e) {
                // Per-channel quantization might fail for some tensor types, that's expected
            }
        }
        
        // Try fake quantization
        try {
            torch::Tensor fake_quantized = torch::fake_quantize_per_tensor_affine(
                input_tensor, scale, zero_point, 0, 255);
        } catch (const std::exception& e) {
            // Fake quantization might fail for some tensor types, that's expected
        }
        
        // Try dynamic quantization
        try {
            torch::Tensor dynamic_quantized = torch::quantize_per_tensor_dynamic(
                input_tensor, q_dtype, false);
        } catch (const std::exception& e) {
            // Dynamic quantization might fail for some tensor types, that's expected
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
