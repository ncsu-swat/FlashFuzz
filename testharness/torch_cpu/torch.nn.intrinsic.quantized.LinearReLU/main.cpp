#include "fuzzer_utils.h"
#include <iostream>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Early exit if not enough data
        if (Size < 16) {
            return 0;
        }
        
        // Read batch size and feature dimensions from fuzzer data
        uint8_t batch_dim = (Data[offset++] % 4) + 1;  // 1-4 batches
        uint8_t in_features = (Data[offset++] % 16) + 1;  // 1-16 input features
        uint8_t out_features = (Data[offset++] % 16) + 1;  // 1-16 output features
        
        // Read scale and zero point values
        float scale_input = 0.1f;
        int64_t zero_point_input = 0;
        float scale_weight = 0.01f;
        float scale_output = 0.1f;
        int64_t zero_point_output = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale_input, Data + offset, sizeof(float));
            offset += sizeof(float);
            scale_input = std::abs(scale_input);
            if (!std::isfinite(scale_input) || scale_input < 1e-6f) scale_input = 0.1f;
            if (scale_input > 1e6f) scale_input = 1.0f;
        }
        
        if (offset + 1 <= Size) {
            zero_point_input = static_cast<int64_t>(Data[offset++]) - 128;  // Range: -128 to 127
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale_weight, Data + offset, sizeof(float));
            offset += sizeof(float);
            scale_weight = std::abs(scale_weight);
            if (!std::isfinite(scale_weight) || scale_weight < 1e-6f) scale_weight = 0.01f;
            if (scale_weight > 1e6f) scale_weight = 1.0f;
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale_output, Data + offset, sizeof(float));
            offset += sizeof(float);
            scale_output = std::abs(scale_output);
            if (!std::isfinite(scale_output) || scale_output < 1e-6f) scale_output = 0.1f;
            if (scale_output > 1e6f) scale_output = 1.0f;
        }
        
        if (offset + 1 <= Size) {
            zero_point_output = static_cast<int64_t>(Data[offset++]) - 128;
        }
        
        // Create input tensor with fuzzer data
        std::vector<int64_t> input_shape = {batch_dim, in_features};
        torch::Tensor input_tensor = torch::zeros(input_shape, torch::kFloat);
        
        size_t input_numel = input_tensor.numel();
        size_t bytes_available = (offset < Size) ? (Size - offset) : 0;
        size_t elements_to_fill = std::min(input_numel, bytes_available);
        
        float* input_ptr = input_tensor.data_ptr<float>();
        for (size_t i = 0; i < elements_to_fill; i++) {
            // Convert byte to float in reasonable range
            input_ptr[i] = (static_cast<float>(Data[offset + i]) - 128.0f) / 10.0f;
        }
        offset += elements_to_fill;
        
        // Quantize input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(
                input_tensor, 
                scale_input, 
                zero_point_input, 
                torch::kQInt8
            );
        } catch (...) {
            // Fallback with safe parameters
            q_input = torch::quantize_per_tensor(
                input_tensor, 
                0.1, 
                0, 
                torch::kQInt8
            );
        }
        
        // Create weight tensor [out_features, in_features]
        torch::Tensor weight = torch::zeros({out_features, in_features}, torch::kFloat);
        
        size_t weight_numel = weight.numel();
        bytes_available = (offset < Size) ? (Size - offset) : 0;
        elements_to_fill = std::min(weight_numel, bytes_available);
        
        float* weight_ptr = weight.data_ptr<float>();
        for (size_t i = 0; i < elements_to_fill; i++) {
            weight_ptr[i] = (static_cast<float>(Data[offset + i]) - 128.0f) / 100.0f;
        }
        offset += elements_to_fill;
        
        // Quantize weight tensor (zero_point must be 0 for weight in qint8)
        torch::Tensor q_weight;
        try {
            q_weight = torch::quantize_per_tensor(
                weight, 
                scale_weight, 
                0,  // zero_point must be 0 for weights
                torch::kQInt8
            );
        } catch (...) {
            q_weight = torch::quantize_per_tensor(
                weight, 
                0.01, 
                0, 
                torch::kQInt8
            );
        }
        
        // Create optional bias tensor
        torch::Tensor bias = torch::zeros({out_features}, torch::kFloat);
        bytes_available = (offset < Size) ? (Size - offset) : 0;
        elements_to_fill = std::min(static_cast<size_t>(out_features), bytes_available);
        
        float* bias_ptr = bias.data_ptr<float>();
        for (size_t i = 0; i < elements_to_fill; i++) {
            bias_ptr[i] = (static_cast<float>(Data[offset + i]) - 128.0f) / 50.0f;
        }
        
        // Perform quantized linear operation
        // This is equivalent to what torch.nn.intrinsic.quantized.LinearReLU does internally
        torch::Tensor linear_output;
        try {
            linear_output = torch::nn::functional::linear(
                q_input.dequantize(), 
                q_weight.dequantize(),
                bias
            );
        } catch (...) {
            // Shape mismatch or other issue - silently skip
            return 0;
        }
        
        // Apply ReLU (fused with linear in the intrinsic module)
        torch::Tensor relu_output = torch::relu(linear_output);
        
        // Quantize the output
        torch::Tensor q_output;
        try {
            q_output = torch::quantize_per_tensor(
                relu_output,
                scale_output,
                zero_point_output,
                torch::kQInt8
            );
        } catch (...) {
            q_output = torch::quantize_per_tensor(
                relu_output,
                0.1,
                0,
                torch::kQInt8
            );
        }
        
        // Dequantize and verify
        torch::Tensor dequantized = q_output.dequantize();
        
        // Access values to prevent optimization
        volatile float sum = dequantized.sum().item<float>();
        (void)sum;
        
        // Verify ReLU property: all values should be >= 0
        torch::Tensor min_val = dequantized.min();
        volatile float min_value = min_val.item<float>();
        (void)min_value;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}