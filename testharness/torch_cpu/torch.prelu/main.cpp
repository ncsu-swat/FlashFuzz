#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight tensor for PReLU
        // PReLU weight should be a 1D tensor with size equal to number of channels
        // or a single scalar for channel-shared case
        torch::Tensor weight;
        
        // Decide between scalar weight and per-channel weight
        if (offset < Size) {
            uint8_t weight_type = Data[offset++];
            
            if (weight_type % 2 == 0) {
                // Scalar weight case
                if (offset < Size) {
                    float scalar_value = static_cast<float>(Data[offset++]) / 255.0f;
                    weight = torch::tensor(scalar_value);
                } else {
                    weight = torch::tensor(0.25f);
                }
            } else {
                // Per-channel weight case
                // For per-channel, we need a tensor with size equal to number of channels
                // For N-D tensor, channels are typically at dimension 1 (0-indexed)
                int64_t num_channels = 1;
                if (input.dim() > 1 && input.size(1) > 0) {
                    num_channels = input.size(1);
                }
                
                // Create weight tensor with appropriate size
                std::vector<float> weight_data;
                for (int64_t i = 0; i < num_channels && offset < Size; i++) {
                    float value = static_cast<float>(Data[offset++]) / 255.0f;
                    weight_data.push_back(value);
                }
                
                // If we don't have enough data, fill the rest with a default value
                while (weight_data.size() < static_cast<size_t>(num_channels)) {
                    weight_data.push_back(0.25f);
                }
                
                weight = torch::tensor(weight_data);
            }
        } else {
            // Default weight if we don't have enough data
            weight = torch::tensor(0.25f);
        }
        
        // Apply PReLU operation
        torch::Tensor output = torch::prelu(input, weight);
        
        // Optionally test some edge cases
        if (offset < Size && Data[offset++] % 3 == 0) {
            // Test with negative input values to ensure PReLU behavior
            torch::Tensor negative_input = -torch::abs(input);
            torch::Tensor negative_output = torch::prelu(negative_input, weight);
        }
        
        // Test with different weight shapes if input has channels
        if (input.dim() > 1 && input.size(1) > 1 && offset < Size && Data[offset++] % 2 == 0) {
            // Test with scalar weight (channel-shared)
            torch::Tensor scalar_weight = torch::tensor(0.5f);
            torch::Tensor output_scalar = torch::prelu(input, scalar_weight);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}