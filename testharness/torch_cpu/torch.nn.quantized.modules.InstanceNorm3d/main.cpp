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
        
        // Early return if not enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor - must be 5D for InstanceNorm3d (N, C, D, H, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is 5D for InstanceNorm3d
        if (input.dim() != 5) {
            // Reshape to 5D if needed
            std::vector<int64_t> new_shape;
            if (input.dim() < 5) {
                // Expand dimensions to 5D
                new_shape = input.sizes().vec();
                while (new_shape.size() < 5) {
                    new_shape.push_back(1);
                }
            } else if (input.dim() > 5) {
                // Collapse extra dimensions
                new_shape.push_back(input.size(0)); // N
                new_shape.push_back(input.size(1)); // C
                
                // Combine remaining dimensions into D, H, W
                int64_t d_size = 1;
                for (int i = 2; i < input.dim() - 2; ++i) {
                    d_size *= input.size(i);
                }
                new_shape.push_back(d_size); // D
                new_shape.push_back(input.size(input.dim() - 2)); // H
                new_shape.push_back(input.size(input.dim() - 1)); // W
            }
            
            // Apply reshape if needed
            if (!new_shape.empty()) {
                input = input.reshape(new_shape);
            }
        }
        
        // Ensure we have at least one channel
        if (input.size(1) == 0) {
            input = input.reshape({input.size(0), 1, input.size(2), input.size(3), input.size(4)});
        }
        
        // Convert to float for quantization
        if (input.scalar_type() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Get parameters for InstanceNorm3d from the input data
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        if (offset + 4 <= Size) {
            // Extract parameters from input data
            eps = std::abs(*reinterpret_cast<const float*>(Data + offset)) / 1000.0 + 1e-10;
            offset += sizeof(float);
            
            momentum = std::abs(*reinterpret_cast<const float*>(Data + offset)) / 10.0;
            momentum = std::min(momentum, 1.0);
            offset += sizeof(float);
            
            affine = (Data[offset++] % 2) == 0;
            track_running_stats = (Data[offset++] % 2) == 0;
        }
        
        // Get scale and zero_point for quantization
        float scale = 0.1;
        int zero_point = 0;
        
        if (offset + sizeof(float) + sizeof(int) <= Size) {
            scale = std::abs(*reinterpret_cast<const float*>(Data + offset)) + 1e-10;
            offset += sizeof(float);
            
            zero_point = *reinterpret_cast<const int*>(Data + offset) % 256;
            offset += sizeof(int);
        }
        
        // Create quantized tensor
        auto qscheme = torch::kPerTensorAffine;
        auto qtype = torch::kQUInt8;
        
        // Quantize the input tensor
        torch::Tensor q_input = torch::quantize_per_tensor(input, scale, zero_point, qtype);
        
        // Create InstanceNorm3d module
        int64_t num_features = q_input.size(1);
        torch::nn::InstanceNorm3d instance_norm(
            torch::nn::InstanceNorm3dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Since torch::nn::quantized::InstanceNorm3d doesn't exist in PyTorch C++ API,
        // we'll use the regular InstanceNorm3d with quantized input
        // Forward pass with quantized input (dequantize first)
        torch::Tensor dequantized_input = q_input.dequantize();
        torch::Tensor output = instance_norm(dequantized_input);
        
        // Quantize the output
        torch::Tensor q_output = torch::quantize_per_tensor(output, scale, zero_point, qtype);
        
        // Try to access properties and methods
        auto options = instance_norm->options;
        
        // Try dequantizing the output
        torch::Tensor final_dequantized = q_output.dequantize();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
