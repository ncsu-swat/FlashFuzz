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
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception &e) {
            return 0;
        }
        
        // Ensure we have at least 4 bytes left for parameters
        if (Size - offset < 4) {
            return 0;
        }
        
        // Extract parameters for BatchNorm2d
        int64_t num_features = 0;
        if (input_tensor.dim() >= 2) {
            num_features = input_tensor.size(1); // Use channel dimension
        } else if (input_tensor.dim() == 1 && input_tensor.size(0) > 0) {
            num_features = input_tensor.size(0);
        } else {
            num_features = 1 + (Data[offset++] % 64); // Random number of features
        }
        
        // Extract eps parameter (small positive value for numerical stability)
        double eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure eps is positive but not too large
            eps = std::abs(eps_raw);
            if (eps == 0.0) eps = 1e-5;
            if (eps > 0.1) eps = 0.1;
        }
        
        // Extract momentum parameter
        double momentum = 0.1;
        if (offset + sizeof(float) <= Size) {
            float momentum_raw;
            std::memcpy(&momentum_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Clamp momentum to [0, 1]
            momentum = std::abs(momentum_raw);
            if (momentum > 1.0) momentum = 1.0;
        }
        
        // Create regular BatchNorm2d module (quantized version not available in C++ frontend)
        torch::nn::BatchNorm2d bn(torch::nn::BatchNorm2dOptions(num_features)
                                  .eps(eps)
                                  .momentum(momentum));
        
        // Ensure input tensor has correct shape and type for BatchNorm2d
        if (input_tensor.dim() < 2) {
            // Reshape to [N, C, H, W] format
            input_tensor = input_tensor.reshape({1, num_features, 1, 1});
        } else if (input_tensor.dim() == 2) {
            // Add H and W dimensions
            input_tensor = input_tensor.unsqueeze(2).unsqueeze(3);
        } else if (input_tensor.dim() == 3) {
            // Add W dimension
            input_tensor = input_tensor.unsqueeze(3);
        }
        
        // Ensure channel dimension matches num_features
        if (input_tensor.size(1) != num_features) {
            input_tensor = input_tensor.transpose(0, 1);
            if (input_tensor.size(1) != num_features) {
                // Reshape to ensure channel dimension is correct
                std::vector<int64_t> new_shape = {input_tensor.size(0), num_features};
                for (int i = 2; i < input_tensor.dim(); i++) {
                    new_shape.push_back(input_tensor.size(i));
                }
                input_tensor = input_tensor.reshape(new_shape);
            }
        }
        
        // Quantize the input tensor
        auto scale = 1.0f / 128.0f;
        auto zero_point = 128;
        
        // Create a quantized tensor
        torch::Tensor quantized_input;
        try {
            quantized_input = torch::quantize_per_tensor(
                input_tensor.to(torch::kFloat),
                scale,
                zero_point,
                torch::kQUInt8
            );
        } catch (const std::exception &e) {
            // If quantization fails, try with a different tensor
            input_tensor = torch::rand({1, num_features, 2, 2});
            quantized_input = torch::quantize_per_tensor(
                input_tensor,
                scale,
                zero_point,
                torch::kQUInt8
            );
        }
        
        // Apply BatchNorm2d on dequantized input, then quantize output
        torch::Tensor output;
        try {
            torch::Tensor dequantized_input = quantized_input.dequantize();
            torch::Tensor bn_output = bn(dequantized_input);
            output = torch::quantize_per_tensor(bn_output, scale, zero_point, torch::kQUInt8);
        } catch (const std::exception &e) {
            // If forward pass fails, try with default parameters
            bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_features));
            torch::Tensor dequantized_input = quantized_input.dequantize();
            torch::Tensor bn_output = bn(dequantized_input);
            output = torch::quantize_per_tensor(bn_output, scale, zero_point, torch::kQUInt8);
        }
        
        // Test additional operations on the output
        if (offset < Size) {
            uint8_t op_selector = Data[offset++];
            
            switch (op_selector % 3) {
                case 0:
                    // Test dequantization
                    try {
                        torch::Tensor dequantized = output.dequantize();
                    } catch (...) {}
                    break;
                    
                case 1:
                    // Test concatenation with another tensor
                    try {
                        auto other_quantized = torch::quantize_per_tensor(
                            torch::rand({1, num_features, 2, 2}),
                            scale,
                            zero_point,
                            torch::kQUInt8
                        );
                        auto cat_dim = output.dim() > 0 ? (Data[offset % Size] % output.dim()) : 0;
                        torch::Tensor cat_result = torch::cat({output, other_quantized}, cat_dim);
                    } catch (...) {}
                    break;
                    
                case 2:
                    // Test getting scale and zero_point
                    try {
                        double out_scale = output.q_scale();
                        int64_t out_zero_point = output.q_zero_point();
                    } catch (...) {}
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
