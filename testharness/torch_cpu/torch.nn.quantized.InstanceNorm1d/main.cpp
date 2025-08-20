#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Parse parameters for InstanceNorm1d
        int64_t num_features = 0;
        float eps = 1e-5;
        float momentum = 0.1;
        bool affine = false;
        bool track_running_stats = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&num_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_features = std::abs(num_features) % 100 + 1;
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (!std::isfinite(eps) || eps <= 0) {
                eps = 1e-5;
            }
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (!std::isfinite(momentum) || momentum < 0 || momentum > 1) {
                momentum = 0.1;
            }
        }
        
        if (offset < Size) {
            affine = Data[offset++] & 1;
        }
        
        if (offset < Size) {
            track_running_stats = Data[offset++] & 1;
        }
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Ensure input tensor is 3D for InstanceNorm1d (N, C, L)
        if (input.dim() != 3) {
            // Reshape to 3D if needed
            int64_t total_elements = input.numel();
            if (total_elements > 0) {
                int64_t batch_size = 1;
                int64_t length = std::max<int64_t>(1, total_elements / (batch_size * num_features));
                input = input.reshape({batch_size, num_features, length});
            } else {
                input = torch::zeros({1, num_features, 1});
            }
        }
        
        // Quantize the input tensor
        auto scale = 1.0f / 128.0f;
        auto zero_point = 128;
        auto qtype = torch::kQUInt8;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (!std::isfinite(scale) || scale <= 0) {
                scale = 1.0f / 128.0f;
            }
        }
        
        if (offset + sizeof(int) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int));
            offset += sizeof(int);
            zero_point = std::abs(zero_point) % 256;
        }
        
        // Convert to float first to ensure compatibility
        input = input.to(torch::kFloat);
        
        // Quantize the input tensor
        auto q_input = torch::quantize_per_tensor(input, scale, zero_point, qtype);
        
        // Create regular InstanceNorm1d and apply quantized instance normalization manually
        auto options = torch::nn::InstanceNorm1dOptions(num_features)
                           .eps(eps)
                           .momentum(momentum)
                           .affine(affine)
                           .track_running_stats(track_running_stats);
        
        auto instance_norm = torch::nn::InstanceNorm1d(options);
        
        // Dequantize, apply instance norm, then quantize back
        auto dequantized_input = q_input.dequantize();
        auto output_float = instance_norm(dequantized_input);
        auto output = torch::quantize_per_tensor(output_float, scale, zero_point, qtype);
        
        // Dequantize to verify the result
        auto dequantized = output.dequantize();
        
        // Perform some operations on the output to ensure it's used
        auto sum = dequantized.sum();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}