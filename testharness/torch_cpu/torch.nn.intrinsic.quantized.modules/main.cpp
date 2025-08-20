#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a byte to determine which quantized module to test
        uint8_t module_selector = 0;
        if (offset < Size) {
            module_selector = Data[offset++];
        }
        
        // Create scale and zero_point for quantization
        double scale = 0.1;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            scale = std::abs(scale) + 1e-5; // Ensure positive scale
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            zero_point = zero_point % 256; // Keep zero_point in uint8 range
        }
        
        // Get parameters for modules
        int64_t in_channels = 3;
        int64_t out_channels = 3;
        int64_t kernel_size = 3;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t groups = 1;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&in_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            in_channels = std::abs(in_channels) % 16 + 1; // Ensure positive
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_channels = std::abs(out_channels) % 16 + 1; // Ensure positive
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_size = std::abs(kernel_size) % 5 + 1; // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride = std::abs(stride) % 3 + 1; // Ensure positive
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding = std::abs(padding) % 3; // Ensure non-negative
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            groups = std::abs(groups) % in_channels + 1; // Ensure positive and valid
            if (in_channels % groups != 0) {
                groups = 1; // Ensure in_channels is divisible by groups
            }
        }
        
        // Reshape input tensor if needed for convolution
        if (input_tensor.dim() < 3) {
            // For Conv2d, we need at least 3 dimensions (N, C, H, W)
            std::vector<int64_t> new_shape = {1, in_channels, kernel_size, kernel_size};
            input_tensor = input_tensor.reshape(new_shape);
        } else if (input_tensor.size(1) != in_channels) {
            // Ensure channel dimension matches in_channels
            std::vector<int64_t> new_shape = input_tensor.sizes().vec();
            new_shape[1] = in_channels;
            input_tensor = input_tensor.reshape(new_shape);
        }
        
        // Quantize the input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(
                input_tensor.to(torch::kFloat), 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
        } catch (...) {
            // If quantization fails, create a basic quantized tensor
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            auto float_tensor = torch::ones({1, in_channels, kernel_size, kernel_size}, options);
            q_input = torch::quantize_per_tensor(float_tensor, scale, zero_point, torch::kQUInt8);
        }
        
        // Test different quantized operations based on the selector
        switch (module_selector % 5) {
            case 0: {
                // Test quantized convolution
                auto weight = torch::randn({out_channels, in_channels / groups, kernel_size, kernel_size});
                auto q_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQInt8);
                
                try {
                    auto output = torch::conv2d(q_input, q_weight, {}, stride, padding, 1, groups);
                } catch (...) {
                    // Ignore exceptions from the operation execution
                }
                break;
            }
            case 1: {
                // Test quantized linear
                auto weight = torch::randn({out_channels, in_channels});
                auto q_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQInt8);
                
                // Reshape for linear layer if needed
                auto linear_input = q_input;
                if (linear_input.dim() > 2) {
                    linear_input = linear_input.reshape({linear_input.size(0), -1});
                }
                
                try {
                    auto output = torch::linear(linear_input, q_weight);
                } catch (...) {
                    // Ignore exceptions from the operation execution
                }
                break;
            }
            case 2: {
                // Test quantized relu
                try {
                    auto output = torch::relu(q_input);
                } catch (...) {
                    // Ignore exceptions from the operation execution
                }
                break;
            }
            case 3: {
                // Test quantized add
                auto other = torch::quantize_per_tensor(
                    torch::ones_like(input_tensor.to(torch::kFloat)), 
                    scale, 
                    zero_point, 
                    torch::kQUInt8
                );
                
                try {
                    auto output = torch::add(q_input, other);
                } catch (...) {
                    // Ignore exceptions from the operation execution
                }
                break;
            }
            case 4: {
                // Test dequantization
                try {
                    auto output = torch::dequantize(q_input);
                } catch (...) {
                    // Ignore exceptions from the operation execution
                }
                break;
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