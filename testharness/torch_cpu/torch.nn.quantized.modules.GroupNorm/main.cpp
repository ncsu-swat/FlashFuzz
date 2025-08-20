#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for GroupNorm
        // We need at least 4 bytes for the parameters
        if (offset + 4 > Size) {
            return 0;
        }
        
        // Extract number of groups
        uint16_t num_groups_raw;
        std::memcpy(&num_groups_raw, Data + offset, sizeof(uint16_t));
        offset += sizeof(uint16_t);
        
        // Ensure num_groups is at least 1 and not too large
        int64_t num_groups = (num_groups_raw % 64) + 1;
        
        // Extract number of channels
        uint16_t num_channels_raw;
        std::memcpy(&num_channels_raw, Data + offset, sizeof(uint16_t));
        offset += sizeof(uint16_t);
        
        // Ensure num_channels is at least num_groups and not too large
        int64_t num_channels = num_groups * ((num_channels_raw % 16) + 1);
        
        // Extract epsilon
        double eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            float eps_float;
            std::memcpy(&eps_float, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure epsilon is positive
            eps = std::abs(eps_float);
            if (eps == 0.0) eps = 1e-5;
        }
        
        // Extract affine flag
        bool affine = true;
        if (offset < Size) {
            affine = (Data[offset++] & 0x01) != 0;
        }
        
        // Create a quantized tensor if the input is not already quantized
        torch::Tensor quantized_input;
        if (!input_tensor.is_quantized()) {
            // Reshape input tensor to have the right number of channels if needed
            std::vector<int64_t> input_shape = input_tensor.sizes().vec();
            if (input_shape.size() >= 2) {
                input_shape[1] = num_channels;
                input_tensor = input_tensor.reshape(input_shape);
            } else if (input_shape.size() == 1) {
                // Add batch dimension and channel dimension
                input_tensor = input_tensor.reshape({1, num_channels, -1});
            } else if (input_shape.size() == 0) {
                // Create a minimal tensor with batch and channel dimensions
                input_tensor = torch::ones({1, num_channels, 1});
            }
            
            // Quantize the tensor
            float scale = 0.1f;
            int zero_point = 0;
            quantized_input = torch::quantize_per_tensor(
                input_tensor.to(torch::kFloat), 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
        } else {
            quantized_input = input_tensor;
        }
        
        // Create weight and bias tensors if affine is true
        torch::Tensor weight, bias;
        if (affine) {
            weight = torch::ones({num_channels});
            bias = torch::zeros({num_channels});
        }
        
        // Apply GroupNorm using the functional interface
        torch::Tensor output = torch::group_norm(
            quantized_input.dequantize(),
            num_groups,
            affine ? c10::optional<torch::Tensor>(weight) : c10::nullopt,
            affine ? c10::optional<torch::Tensor>(bias) : c10::nullopt,
            eps
        );
        
        // Ensure the output is valid
        if (output.numel() > 0) {
            // Access some elements to ensure computation happened
            float sum = output.sum().item<float>();
            (void)sum;  // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}