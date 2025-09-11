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
        
        // Early exit if not enough data
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get dimensions for weight matrix
        int64_t in_features = 0;
        if (input_tensor.dim() > 0) {
            in_features = input_tensor.size(-1);
        } else {
            in_features = 1; // For scalar input
        }
        
        // Determine out_features from remaining data
        int64_t out_features = 4; // Default
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure out_features is reasonable but allow edge cases
            out_features = std::abs(out_features) % 32 + 1;
        }
        
        // Create scale and zero_point for quantization
        double scale_input = 0.1;
        int64_t zero_point_input = 10;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale_input, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure scale is positive and not too extreme
            scale_input = std::abs(scale_input);
            if (scale_input < 1e-10) scale_input = 0.1;
            if (scale_input > 1e10) scale_input = 1.0;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point_input, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure zero_point is within valid range for int8
            zero_point_input = zero_point_input % 256;
        }
        
        // Create quantized input tensor
        torch::Tensor q_input;
        try {
            // Quantize the input tensor
            q_input = torch::quantize_per_tensor(
                input_tensor.to(torch::kFloat), 
                scale_input, 
                zero_point_input, 
                torch::kQInt8
            );
        } catch (const std::exception& e) {
            // If quantization fails, create a simple quantized tensor
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            auto simple_tensor = torch::ones({std::max(int64_t(1), in_features)}, options);
            q_input = torch::quantize_per_tensor(simple_tensor, 0.1, 10, torch::kQInt8);
        }
        
        // Create weight scale and zero point
        double scale_weight = 0.01;
        int64_t zero_point_weight = 0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale_weight, Data + offset, sizeof(double));
            offset += sizeof(double);
            scale_weight = std::abs(scale_weight);
            if (scale_weight < 1e-10) scale_weight = 0.01;
            if (scale_weight > 1e10) scale_weight = 1.0;
        }
        
        // Create output scale and zero point
        double scale_output = 0.1;
        int64_t zero_point_output = 5;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale_output, Data + offset, sizeof(double));
            offset += sizeof(double);
            scale_output = std::abs(scale_output);
            if (scale_output < 1e-10) scale_output = 0.1;
            if (scale_output > 1e10) scale_output = 1.0;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point_output, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            zero_point_output = zero_point_output % 256;
        }
        
        // Create weight tensor
        torch::Tensor weight = torch::ones({out_features, in_features}, torch::kFloat);
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t weight_size = weight.numel() * sizeof(float);
            size_t bytes_to_copy = std::min(remaining, weight_size);
            
            if (bytes_to_copy > 0) {
                std::memcpy(weight.data_ptr(), Data + offset, bytes_to_copy);
            }
        }
        
        // Quantize the weight tensor
        torch::Tensor q_weight = torch::quantize_per_tensor(
            weight, 
            scale_weight, 
            zero_point_weight, 
            torch::kQInt8
        );
        
        // Use functional API for quantized linear + relu
        torch::Tensor linear_output = torch::nn::functional::linear(
            q_input.dequantize(), 
            q_weight.dequantize()
        );
        
        // Apply ReLU
        torch::Tensor relu_output = torch::relu(linear_output);
        
        // Quantize the output
        torch::Tensor output = torch::quantize_per_tensor(
            relu_output,
            scale_output,
            zero_point_output,
            torch::kQInt8
        );
        
        // Dequantize to check values
        torch::Tensor dequantized = output.dequantize();
        
        // Access some values to ensure computation is not optimized away
        float sum = dequantized.sum().item<float>();
        if (std::isnan(sum) || std::isinf(sum)) {
            throw std::runtime_error("Output contains NaN or Inf values");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
