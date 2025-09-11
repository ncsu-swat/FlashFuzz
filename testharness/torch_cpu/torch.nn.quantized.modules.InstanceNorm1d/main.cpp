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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the input tensor has at least 2 dimensions for InstanceNorm1d
        // (batch_size, channels, length)
        if (input_tensor.dim() < 2) {
            input_tensor = input_tensor.reshape({1, input_tensor.numel()});
        }
        
        // If we have a 2D tensor, add a third dimension
        if (input_tensor.dim() == 2) {
            input_tensor = input_tensor.unsqueeze(-1);
        }
        
        // Ensure we have a 3D tensor with proper shape for InstanceNorm1d
        if (input_tensor.dim() > 3) {
            input_tensor = input_tensor.flatten(2).slice(2, 0, 1);
        }
        
        // Get parameters for InstanceNorm1d from the input data
        uint8_t num_features_byte = (offset < Size) ? Data[offset++] : 0;
        int64_t num_features = std::max(int64_t(1), int64_t(num_features_byte));
        
        // Ensure the tensor has the right shape for InstanceNorm1d
        // Reshape to [batch_size, num_features, length]
        auto batch_size = input_tensor.size(0);
        auto length = (input_tensor.dim() > 2) ? input_tensor.size(2) : 1;
        
        // Reshape the input tensor to match the required dimensions
        input_tensor = input_tensor.reshape({batch_size, num_features, length});
        
        // Get additional parameters
        double eps = 1e-5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-5;
        }
        
        double momentum = 0.1;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is between 0 and 1
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        bool affine = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        bool track_running_stats = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        
        // Quantization parameters
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            scale = std::abs(scale);
            if (scale == 0.0) scale = 1.0;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Convert to quantized tensor if needed
        torch::Tensor quantized_input;
        
        // Create a quantized tensor
        auto qscheme = at::kPerTensorAffine;
        if (offset < Size && Data[offset++] % 2 == 0) {
            qscheme = at::kPerTensorSymmetric;
        }
        
        // Quantize the input tensor
        quantized_input = torch::quantize_per_tensor(
            input_tensor.to(torch::kFloat), 
            scale, 
            zero_point, 
            torch::kQUInt8
        );
        
        // Create InstanceNorm1d module
        torch::nn::InstanceNorm1d instance_norm(
            torch::nn::InstanceNorm1dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the InstanceNorm1d to quantized input
        torch::Tensor output;
        try {
            // Dequantize, apply instance norm, then quantize back
            torch::Tensor dequantized_input = quantized_input.dequantize();
            torch::Tensor norm_output = instance_norm->forward(dequantized_input);
            output = torch::quantize_per_tensor(norm_output, scale, zero_point, torch::kQUInt8);
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
            return 0;
        }
        
        // Try to access properties of the output to ensure it's valid
        auto output_sizes = output.sizes();
        auto output_dtype = output.dtype();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
