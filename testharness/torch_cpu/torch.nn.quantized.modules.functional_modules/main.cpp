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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create scale and zero_point for quantization
        double scale = 0.1;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) + sizeof(int64_t) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure scale is positive and reasonable
        scale = std::abs(scale);
        if (scale < 1e-10) scale = 0.1;
        
        // Constrain zero_point to valid range for int8
        zero_point = std::max(std::min(zero_point, static_cast<int64_t>(127)), static_cast<int64_t>(-128));
        
        // Create quantized tensor
        torch::Tensor quantized;
        try {
            quantized = torch::quantize_per_tensor(input, scale, zero_point, torch::kQInt8);
        } catch (const std::exception& e) {
            // If quantization fails, try with float tensor
            if (input.dtype() != torch::kFloat) {
                input = input.to(torch::kFloat);
                quantized = torch::quantize_per_tensor(input, scale, zero_point, torch::kQInt8);
            } else {
                throw;
            }
        }
        
        // Test various functional modules from torch.nn.quantized.modules.functional_modules
        
        // 1. Test quantized::relu
        try {
            auto relu_result = torch::nn::functional::relu(quantized);
        } catch (...) {}
        
        // 2. Test quantized::linear
        try {
            // Create weight and bias tensors
            torch::Tensor weight;
            torch::Tensor bias;
            
            if (offset < Size) {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure weight has compatible shape for linear operation
                if (input.dim() > 0 && weight.dim() > 1) {
                    int64_t input_features = input.size(-1);
                    weight = weight.reshape({weight.size(0), input_features});
                    
                    // Create bias if there's still data
                    if (offset < Size) {
                        bias = fuzzer_utils::createTensor(Data, Size, offset);
                        if (bias.dim() > 0) {
                            bias = bias.reshape({weight.size(0)});
                        }
                    }
                    
                    // Quantize weight and bias
                    auto q_weight = torch::quantize_per_tensor(weight.to(torch::kFloat), scale, zero_point, torch::kQInt8);
                    
                    // Test linear function
                    auto linear_result = torch::nn::functional::linear(quantized, q_weight, bias);
                }
            }
        } catch (...) {}
        
        // 3. Test quantized::conv2d
        try {
            if (input.dim() >= 4) {
                // Create weight for conv2d
                torch::Tensor weight;
                torch::Tensor bias;
                
                if (offset < Size) {
                    weight = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Ensure weight has compatible shape for conv2d
                    if (weight.dim() >= 4) {
                        int64_t in_channels = input.size(1);
                        weight = weight.reshape({weight.size(0), in_channels, weight.size(2), weight.size(3)});
                        
                        // Create bias if there's still data
                        if (offset < Size) {
                            bias = fuzzer_utils::createTensor(Data, Size, offset);
                            if (bias.dim() > 0) {
                                bias = bias.reshape({weight.size(0)});
                            }
                        }
                        
                        // Parse stride, padding, dilation, groups
                        std::vector<int64_t> stride = {1, 1};
                        std::vector<int64_t> padding = {0, 0};
                        std::vector<int64_t> dilation = {1, 1};
                        int64_t groups = 1;
                        
                        if (offset + 5 * sizeof(int64_t) <= Size) {
                            std::memcpy(&stride[0], Data + offset, sizeof(int64_t));
                            offset += sizeof(int64_t);
                            std::memcpy(&stride[1], Data + offset, sizeof(int64_t));
                            offset += sizeof(int64_t);
                            
                            std::memcpy(&padding[0], Data + offset, sizeof(int64_t));
                            offset += sizeof(int64_t);
                            std::memcpy(&padding[1], Data + offset, sizeof(int64_t));
                            offset += sizeof(int64_t);
                            
                            std::memcpy(&groups, Data + offset, sizeof(int64_t));
                            offset += sizeof(int64_t);
                            
                            // Ensure values are reasonable
                            stride[0] = std::abs(stride[0]) % 5 + 1;
                            stride[1] = std::abs(stride[1]) % 5 + 1;
                            padding[0] = std::abs(padding[0]) % 3;
                            padding[1] = std::abs(padding[1]) % 3;
                            groups = std::max(int64_t(1), std::min(groups, in_channels));
                        }
                        
                        // Quantize weight
                        auto q_weight = torch::quantize_per_tensor(weight.to(torch::kFloat), scale, zero_point, torch::kQInt8);
                        
                        // Test conv2d function using at::conv2d
                        auto conv_result = at::conv2d(
                            quantized, q_weight, bias, stride, padding, dilation, groups);
                    }
                }
            }
        } catch (...) {}
        
        // 4. Test quantized::max_pool2d
        try {
            if (input.dim() >= 4) {
                // Parse kernel_size, stride, padding
                std::vector<int64_t> kernel_size = {2, 2};
                std::vector<int64_t> stride = {1, 1};
                std::vector<int64_t> padding = {0, 0};
                
                if (offset + 6 * sizeof(int64_t) <= Size) {
                    std::memcpy(&kernel_size[0], Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    std::memcpy(&kernel_size[1], Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    
                    std::memcpy(&stride[0], Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    std::memcpy(&stride[1], Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    
                    std::memcpy(&padding[0], Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    std::memcpy(&padding[1], Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    
                    // Ensure values are reasonable
                    kernel_size[0] = std::abs(kernel_size[0]) % 5 + 1;
                    kernel_size[1] = std::abs(kernel_size[1]) % 5 + 1;
                    stride[0] = std::abs(stride[0]) % 5 + 1;
                    stride[1] = std::abs(stride[1]) % 5 + 1;
                    padding[0] = std::abs(padding[0]) % 3;
                    padding[1] = std::abs(padding[1]) % 3;
                }
                
                // Test max_pool2d function using at::max_pool2d
                auto pool_result = at::max_pool2d(
                    quantized, kernel_size, stride, padding);
            }
        } catch (...) {}
        
        // 5. Test quantized::avg_pool2d
        try {
            if (input.dim() >= 4) {
                // Use the same parameters as max_pool2d
                std::vector<int64_t> kernel_size = {2, 2};
                std::vector<int64_t> stride = {1, 1};
                std::vector<int64_t> padding = {0, 0};
                
                if (offset + 6 * sizeof(int64_t) <= Size) {
                    std::memcpy(&kernel_size[0], Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    std::memcpy(&kernel_size[1], Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    
                    std::memcpy(&stride[0], Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    std::memcpy(&stride[1], Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    
                    std::memcpy(&padding[0], Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    std::memcpy(&padding[1], Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    
                    // Ensure values are reasonable
                    kernel_size[0] = std::abs(kernel_size[0]) % 5 + 1;
                    kernel_size[1] = std::abs(kernel_size[1]) % 5 + 1;
                    stride[0] = std::abs(stride[0]) % 5 + 1;
                    stride[1] = std::abs(stride[1]) % 5 + 1;
                    padding[0] = std::abs(padding[0]) % 3;
                    padding[1] = std::abs(padding[1]) % 3;
                }
                
                // Test avg_pool2d function using at::avg_pool2d
                auto pool_result = at::avg_pool2d(
                    quantized, kernel_size, stride, padding);
            }
        } catch (...) {}
        
        // 6. Test quantized::adaptive_avg_pool2d
        try {
            if (input.dim() >= 4) {
                // Parse output size
                std::vector<int64_t> output_size = {1, 1};
                
                if (offset + 2 * sizeof(int64_t) <= Size) {
                    std::memcpy(&output_size[0], Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    std::memcpy(&output_size[1], Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    
                    // Ensure values are reasonable
                    output_size[0] = std::abs(output_size[0]) % 8 + 1;
                    output_size[1] = std::abs(output_size[1]) % 8 + 1;
                }
                
                // Test adaptive_avg_pool2d function using torch::nn::functional
                auto options = torch::nn::functional::AdaptiveAvgPool2dFuncOptions(torch::ExpandingArrayWithOptionalElem<2>(output_size));
                auto pool_result = torch::nn::functional::adaptive_avg_pool2d(
                    quantized, options);
            }
        } catch (...) {}
        
        // 7. Test dequantize
        try {
            auto dequantized = torch::dequantize(quantized);
        } catch (...) {}
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
