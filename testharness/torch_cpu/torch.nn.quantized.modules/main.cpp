#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get scale and zero_point for quantization
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) + sizeof(int64_t) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure scale is positive and reasonable
        scale = std::abs(scale);
        if (scale < 1e-6f) scale = 1e-6f;
        if (scale > 1e6f) scale = 1e6f;
        
        // Ensure zero_point is within valid range for int8
        zero_point = std::max(std::min(zero_point, static_cast<int64_t>(127)), static_cast<int64_t>(-128));
        
        // Quantize the tensor
        torch::Tensor quantized_tensor;
        try {
            quantized_tensor = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQInt8);
        } catch (const std::exception& e) {
            // If quantization fails, try with a different tensor
            input_tensor = torch::ones({2, 3});
            quantized_tensor = torch::quantize_per_tensor(input_tensor, 0.1, 0, torch::kQInt8);
        }
        
        // Test various quantized operations
        
        // 1. Linear operation using functional API
        int64_t in_features = 3;
        int64_t out_features = 2;
        
        if (offset + 2 * sizeof(int64_t) <= Size) {
            std::memcpy(&in_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure dimensions are reasonable
            in_features = std::abs(in_features) % 100 + 1;
            out_features = std::abs(out_features) % 100 + 1;
        }
        
        try {
            // Create weight and bias tensors for linear operation
            torch::Tensor weight = torch::randn({out_features, in_features});
            torch::Tensor bias = torch::randn({out_features});
            
            // Create appropriate input for linear operation
            torch::Tensor linear_input = torch::ones({1, in_features});
            torch::Tensor quantized_linear_input = torch::quantize_per_tensor(linear_input, scale, zero_point, torch::kQInt8);
            
            // Apply linear operation using functional API
            torch::Tensor linear_output = torch::linear(quantized_linear_input.dequantize(), weight, bias);
            
            // Quantize the output
            torch::Tensor quantized_linear_output = torch::quantize_per_tensor(linear_output, scale, zero_point, torch::kQInt8);
        } catch (const std::exception& e) {
            // Continue with other tests if this one fails
        }
        
        // 2. Conv2d operation using functional API
        int64_t in_channels = 3;
        int64_t out_channels = 2;
        int64_t kernel_size = 3;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        
        if (offset + 7 * sizeof(int64_t) <= Size) {
            std::memcpy(&in_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            std::memcpy(&out_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            std::memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            std::memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            std::memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            std::memcpy(&dilation, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure parameters are reasonable
            in_channels = std::abs(in_channels) % 16 + 1;
            out_channels = std::abs(out_channels) % 16 + 1;
            kernel_size = std::abs(kernel_size) % 7 + 1;
            stride = std::abs(stride) % 3 + 1;
            padding = std::abs(padding) % 3;
            dilation = std::abs(dilation) % 2 + 1;
            groups = std::abs(groups) % in_channels + 1;
            if (in_channels % groups != 0) groups = 1;
        }
        
        try {
            // Create weight tensor for conv2d operation
            torch::Tensor conv_weight = torch::randn({out_channels, in_channels / groups, kernel_size, kernel_size});
            torch::Tensor conv_bias = torch::randn({out_channels});
            
            // Create appropriate input for conv operation
            int64_t input_h = kernel_size + 4;
            int64_t input_w = kernel_size + 4;
            torch::Tensor conv_input = torch::ones({1, in_channels, input_h, input_w});
            torch::Tensor quantized_conv_input = torch::quantize_per_tensor(conv_input, scale, zero_point, torch::kQInt8);
            
            // Apply conv2d operation using functional API
            torch::Tensor conv_output = torch::conv2d(
                quantized_conv_input.dequantize(), 
                conv_weight, 
                conv_bias,
                {stride, stride},
                {padding, padding},
                {dilation, dilation},
                groups
            );
            
            // Quantize the output
            torch::Tensor quantized_conv_output = torch::quantize_per_tensor(conv_output, scale, zero_point, torch::kQInt8);
        } catch (const std::exception& e) {
            // Continue with other tests if this one fails
        }
        
        // 3. MaxPool2d operation using functional API
        try {
            // Create appropriate input for maxpool operation
            int64_t input_h = kernel_size + 4;
            int64_t input_w = kernel_size + 4;
            torch::Tensor maxpool_input = torch::ones({1, in_channels, input_h, input_w});
            torch::Tensor quantized_maxpool_input = torch::quantize_per_tensor(maxpool_input, scale, zero_point, torch::kQInt8);
            
            // Apply maxpool2d operation using functional API
            torch::Tensor maxpool_output = torch::max_pool2d(
                quantized_maxpool_input.dequantize(),
                {kernel_size, kernel_size},
                {stride, stride},
                {padding, padding},
                {dilation, dilation}
            );
            
            // Quantize the output
            torch::Tensor quantized_maxpool_output = torch::quantize_per_tensor(maxpool_output, scale, zero_point, torch::kQInt8);
        } catch (const std::exception& e) {
            // Continue with other tests if this one fails
        }
        
        // 4. ReLU operation
        try {
            // Apply ReLU operation to quantized tensor
            torch::Tensor dequantized = quantized_tensor.dequantize();
            torch::Tensor relu_output = torch::relu(dequantized);
            torch::Tensor quantized_relu_output = torch::quantize_per_tensor(relu_output, scale, zero_point, torch::kQInt8);
        } catch (const std::exception& e) {
            // Continue with other tests if this one fails
        }
        
        // 5. Embedding operation using functional API
        int64_t num_embeddings = 10;
        int64_t embedding_dim = 3;
        
        if (offset + 2 * sizeof(int64_t) <= Size) {
            std::memcpy(&num_embeddings, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            std::memcpy(&embedding_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure parameters are reasonable
            num_embeddings = std::abs(num_embeddings) % 100 + 1;
            embedding_dim = std::abs(embedding_dim) % 50 + 1;
        }
        
        try {
            // Create embedding weight tensor
            torch::Tensor embedding_weight = torch::randn({num_embeddings, embedding_dim});
            
            // Create appropriate input for embedding operation (indices)
            torch::Tensor indices = torch::randint(0, num_embeddings, {5});
            
            // Apply embedding operation using functional API
            torch::Tensor embedding_output = torch::embedding(embedding_weight, indices);
            
            // Quantize the output
            torch::Tensor quantized_embedding_output = torch::quantize_per_tensor(embedding_output, scale, zero_point, torch::kQInt8);
        } catch (const std::exception& e) {
            // Continue with other tests if this one fails
        }
        
        // Test quantization and dequantization operations
        try {
            torch::Tensor dequantized = quantized_tensor.dequantize();
            torch::Tensor requantized = torch::quantize_per_tensor(dequantized, scale * 2, zero_point + 1, torch::kQInt8);
        } catch (const std::exception& e) {
            // Continue if this fails
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
