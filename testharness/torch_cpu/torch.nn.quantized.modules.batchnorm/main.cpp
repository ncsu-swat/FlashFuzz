#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for BatchNorm
        uint8_t num_features_byte = 0;
        if (offset < Size) {
            num_features_byte = Data[offset++];
        }
        
        // Ensure num_features is positive
        int64_t num_features = (num_features_byte % 64) + 1;
        
        // Extract eps parameter
        double eps = 1e-5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-5;
        }
        
        // Extract momentum parameter
        double momentum = 0.1;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is between 0 and 1
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        // Create regular BatchNorm module (quantized BatchNorm is not available in C++ frontend)
        torch::nn::BatchNorm2d batchnorm(
            torch::nn::BatchNorm2dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
        );
        
        // Ensure input tensor has correct shape and type for BatchNorm2d
        if (input.dim() >= 2) {
            // For BatchNorm2d, input should be [N, C, H, W]
            // If input doesn't have enough dimensions, add them
            while (input.dim() < 4) {
                input = input.unsqueeze(-1);
            }
            
            // Ensure channel dimension matches num_features
            if (input.size(1) != num_features) {
                // Resize the tensor to have the correct number of channels
                std::vector<int64_t> new_shape = input.sizes().vec();
                new_shape[1] = num_features;
                input = input.reshape(new_shape);
            }
            
            // Quantize the input tensor
            auto scale = 1.0f / 128.0f;
            auto zero_point = 128;
            
            // Create a quantized tensor
            torch::Tensor quantized_input;
            
            try {
                // Try to quantize the input tensor
                quantized_input = torch::quantize_per_tensor(
                    input.to(torch::kFloat), 
                    scale, 
                    zero_point, 
                    torch::kQUInt8
                );
                
                // Dequantize for regular BatchNorm processing
                torch::Tensor dequantized_input = torch::dequantize(quantized_input);
                
                // Apply the BatchNorm
                torch::Tensor output = batchnorm(dequantized_input);
                
                // Quantize the output again to simulate quantized BatchNorm
                torch::Tensor quantized_output = torch::quantize_per_tensor(
                    output, 
                    scale, 
                    zero_point, 
                    torch::kQUInt8
                );
            } 
            catch (const c10::Error& e) {
                // Catch PyTorch-specific errors
                return 0;
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
