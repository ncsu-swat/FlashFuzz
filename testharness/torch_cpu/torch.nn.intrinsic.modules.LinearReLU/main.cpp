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
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for LinearReLU from the remaining data
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = true;
        
        // Parse in_features
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&in_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Make in_features positive and reasonable
            in_features = std::abs(in_features) % 100 + 1;
        } else {
            in_features = 10; // Default value
        }
        
        // Parse out_features
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Make out_features positive and reasonable
            out_features = std::abs(out_features) % 100 + 1;
        } else {
            out_features = 5; // Default value
        }
        
        // Parse bias flag
        if (offset < Size) {
            bias = Data[offset++] & 0x01; // Use lowest bit to determine bias
        }
        
        // Reshape input tensor if needed to match in_features
        // For LinearReLU, the last dimension should match in_features
        std::vector<int64_t> input_shape = input.sizes().vec();
        
        // If input is empty or scalar, reshape it to a vector
        if (input_shape.empty()) {
            input = input.reshape({1, in_features});
        } else {
            // Ensure the last dimension matches in_features
            input_shape.back() = in_features;
            
            // If tensor has only one dimension, add batch dimension
            if (input_shape.size() == 1) {
                input_shape.insert(input_shape.begin(), 1);
            }
            
            // Try to reshape, but if it fails due to size mismatch, create a new tensor
            try {
                input = input.reshape(input_shape);
            } catch (const std::exception&) {
                // Create a new tensor with the right shape
                input = torch::ones(input_shape, input.options());
            }
        }
        
        // Create Linear module and apply ReLU manually since LinearReLU is not available
        torch::nn::Linear linear_layer(torch::nn::LinearOptions(in_features, out_features).bias(bias));
        
        // Apply the linear layer to the input
        torch::Tensor linear_output = linear_layer->forward(input);
        
        // Apply ReLU activation
        torch::Tensor output = torch::relu(linear_output);
        
        // Verify output shape
        auto output_shape = output.sizes();
        auto input_batch_shape = input.sizes().slice(0, input.dim() - 1);
        
        // Check if output has expected shape
        if (output.dim() != input.dim() || 
            output_shape[output.dim() - 1] != out_features) {
            throw std::runtime_error("Output shape mismatch");
        }
        
        // Verify ReLU behavior (no negative values)
        if (torch::any(output < 0).item<bool>()) {
            throw std::runtime_error("Output contains negative values, ReLU not applied correctly");
        }
        
        // Test edge cases: zero input
        torch::Tensor zero_input = torch::zeros_like(input);
        torch::Tensor zero_linear_output = linear_layer->forward(zero_input);
        torch::Tensor zero_output = torch::relu(zero_linear_output);
        
        // Test with different input types
        if (input.scalar_type() != torch::kFloat) {
            torch::Tensor float_input = input.to(torch::kFloat);
            torch::Tensor float_linear_output = linear_layer->forward(float_input);
            torch::Tensor float_output = torch::relu(float_linear_output);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
