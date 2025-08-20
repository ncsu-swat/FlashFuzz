#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for LinearReLU from the remaining data
        int64_t in_features = 0;
        int64_t out_features = 0;
        
        // Parse in_features and out_features from the remaining data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&in_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make in_features positive and reasonable
            in_features = std::abs(in_features) % 128 + 1;
        } else {
            in_features = 10; // Default value
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make out_features positive and reasonable
            out_features = std::abs(out_features) % 128 + 1;
        } else {
            out_features = 5; // Default value
        }
        
        // Create a regular Linear layer and apply ReLU manually
        torch::nn::Linear linear(in_features, out_features);
        
        // Reshape input tensor if needed to match in_features
        // For Linear, the last dimension should match in_features
        auto input_sizes = input.sizes().vec();
        if (input_sizes.empty()) {
            // Scalar input, reshape to [1, in_features]
            input = input.reshape({1, in_features});
        } else {
            // Ensure the last dimension is in_features
            input_sizes.back() = in_features;
            input = input.reshape(input_sizes);
        }
        
        // Convert input to float
        input = input.to(torch::kFloat);
        
        // Apply the Linear operation followed by ReLU
        torch::Tensor linear_output = linear(input);
        torch::Tensor output = torch::relu(linear_output);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        auto mean = output.mean();
        
        // Access the values to prevent optimization from removing the computations
        float sum_val = sum.item<float>();
        float mean_val = mean.item<float>();
        
        // Use the values in a way that doesn't affect the output but prevents
        // compiler from optimizing away the computations
        if (std::isnan(sum_val + mean_val)) {
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}