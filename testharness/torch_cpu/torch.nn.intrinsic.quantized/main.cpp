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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a quantized tensor from the input tensor
        // First, we need to quantize the tensor
        float scale = 1.0f;
        int zero_point = 0;
        
        // Use remaining bytes to determine scale and zero_point if available
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive and not too small
            if (scale <= 0.0f || !std::isfinite(scale)) {
                scale = 1.0f;
            }
        }
        
        if (offset + sizeof(int) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int));
            offset += sizeof(int);
        }
        
        // Quantize the tensor
        torch::Tensor quantized_tensor;
        try {
            // Convert to float if not already
            if (input_tensor.scalar_type() != torch::kFloat) {
                input_tensor = input_tensor.to(torch::kFloat);
            }
            
            // Quantize the tensor
            quantized_tensor = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQUInt8);
            
            // Test various quantized operations
            
            // 1. Test quantized linear
            if (input_tensor.dim() >= 2) {
                int64_t in_features = input_tensor.size(-1);
                int64_t out_features = std::max<int64_t>(1, in_features / 2);
                
                // Create weight and bias
                torch::Tensor weight = torch::randn({out_features, in_features});
                torch::Tensor bias = torch::randn({out_features});
                
                // Quantize weight
                torch::Tensor q_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQUInt8);
                
                // Test quantized linear using functional API
                auto result = torch::nn::functional::linear(quantized_tensor.dequantize(), q_weight.dequantize(), bias);
            }
            
            // 2. Test quantized conv2d
            if (input_tensor.dim() >= 4) {
                int64_t in_channels = input_tensor.size(1);
                int64_t out_channels = std::max<int64_t>(1, in_channels / 2);
                int64_t kernel_size = 3;
                
                // Create weight and bias
                torch::Tensor weight = torch::randn({out_channels, in_channels, kernel_size, kernel_size});
                torch::Tensor bias = torch::randn({out_channels});
                
                // Quantize weight
                torch::Tensor q_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQUInt8);
                
                // Test quantized conv2d using functional API
                auto result = torch::nn::functional::conv2d(
                    quantized_tensor.dequantize(), 
                    q_weight.dequantize(), 
                    torch::nn::functional::Conv2dFuncOptions().bias(bias).stride(1).padding(1)
                );
            }
            
            // 3. Test quantized conv1d
            if (input_tensor.dim() >= 3) {
                int64_t in_channels = input_tensor.size(1);
                int64_t out_channels = std::max<int64_t>(1, in_channels / 2);
                int64_t kernel_size = 3;
                
                // Create weight and bias
                torch::Tensor weight = torch::randn({out_channels, in_channels, kernel_size});
                torch::Tensor bias = torch::randn({out_channels});
                
                // Quantize weight
                torch::Tensor q_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQUInt8);
                
                // Test quantized conv1d using functional API
                auto result = torch::nn::functional::conv1d(
                    quantized_tensor.dequantize(), 
                    q_weight.dequantize(), 
                    torch::nn::functional::Conv1dFuncOptions().bias(bias).stride(1).padding(1)
                );
            }
            
            // 4. Test quantized max_pool2d
            if (input_tensor.dim() >= 4) {
                auto result = torch::nn::functional::max_pool2d(
                    quantized_tensor.dequantize(),
                    torch::nn::functional::MaxPool2dFuncOptions(2).stride(2)
                );
            }
            
            // 5. Test quantized relu
            auto relu_result = torch::relu(quantized_tensor);
            
            // 6. Test quantized add
            auto add_result = torch::add(quantized_tensor, quantized_tensor);
            
            // 7. Test quantized cat
            std::vector<torch::Tensor> tensors_to_cat = {quantized_tensor, quantized_tensor};
            int64_t dim = 0;
            if (input_tensor.dim() > 0) {
                dim = Data[offset % Size] % input_tensor.dim();
            }
            auto cat_result = torch::cat(tensors_to_cat, dim);
        }
        catch (const c10::Error& e) {
            // PyTorch specific errors are expected and can be ignored
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
