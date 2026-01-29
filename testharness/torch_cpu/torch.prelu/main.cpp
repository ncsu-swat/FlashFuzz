#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is float type for PReLU
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create weight tensor for PReLU
        torch::Tensor weight;
        
        // Decide between scalar weight and per-channel weight
        if (offset < Size) {
            uint8_t weight_type = Data[offset++];
            
            if (weight_type % 2 == 0) {
                // Scalar weight case (single element 1D tensor)
                float scalar_value = 0.25f;
                if (offset < Size) {
                    scalar_value = static_cast<float>(Data[offset++]) / 255.0f;
                }
                weight = torch::tensor({scalar_value}, torch::kFloat32);
            } else {
                // Per-channel weight case
                // For N-D tensor (N >= 2), channels are at dimension 1
                // Weight should be 1D tensor with size equal to number of channels
                int64_t num_channels = 1;
                if (input.dim() >= 2 && input.size(1) > 0) {
                    num_channels = input.size(1);
                }
                
                // Create weight tensor with appropriate size
                std::vector<float> weight_data;
                weight_data.reserve(num_channels);
                for (int64_t i = 0; i < num_channels && offset < Size; i++) {
                    float value = static_cast<float>(Data[offset++]) / 255.0f;
                    weight_data.push_back(value);
                }
                
                // Fill the rest with a default value if needed
                while (static_cast<int64_t>(weight_data.size()) < num_channels) {
                    weight_data.push_back(0.25f);
                }
                
                weight = torch::tensor(weight_data, torch::kFloat32);
            }
        } else {
            // Default weight - single element 1D tensor
            weight = torch::tensor({0.25f}, torch::kFloat32);
        }
        
        // Apply PReLU operation
        try {
            torch::Tensor output = torch::prelu(input, weight);
            
            // Force computation
            output.numel();
        } catch (const c10::Error&) {
            // Shape mismatch or other expected errors - try with scalar weight
            weight = torch::tensor({0.25f}, torch::kFloat32);
            torch::Tensor output = torch::prelu(input, weight);
            output.numel();
        }
        
        // Additional test variations based on remaining data
        if (offset < Size && Data[offset++] % 3 == 0) {
            // Test with negative input values to verify PReLU behavior
            try {
                torch::Tensor negative_input = -torch::abs(input);
                torch::Tensor scalar_weight = torch::tensor({0.1f}, torch::kFloat32);
                torch::Tensor negative_output = torch::prelu(negative_input, scalar_weight);
                negative_output.numel();
            } catch (const c10::Error&) {
                // Ignore shape-related errors
            }
        }
        
        // Test with zero weight
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                torch::Tensor zero_weight = torch::tensor({0.0f}, torch::kFloat32);
                torch::Tensor zero_output = torch::prelu(input, zero_weight);
                zero_output.numel();
            } catch (const c10::Error&) {
                // Ignore errors
            }
        }
        
        // Test with larger weight values
        if (offset < Size && Data[offset++] % 5 == 0) {
            try {
                float large_val = static_cast<float>(Data[offset % Size]) / 25.5f; // 0-10 range
                torch::Tensor large_weight = torch::tensor({large_val}, torch::kFloat32);
                torch::Tensor large_output = torch::prelu(input, large_weight);
                large_output.numel();
            } catch (const c10::Error&) {
                // Ignore errors
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}