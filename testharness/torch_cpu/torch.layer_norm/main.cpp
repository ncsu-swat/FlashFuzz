#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract normalized_shape from the input tensor
        std::vector<int64_t> normalized_shape;
        if (input.dim() > 0) {
            int64_t last_dims = std::min(static_cast<int64_t>(3), input.dim());
            
            // Use the last 'last_dims' dimensions as normalized_shape
            for (int64_t i = input.dim() - last_dims; i < input.dim(); i++) {
                normalized_shape.push_back(input.size(i));
            }
        } else {
            // For scalar tensors, use a default shape
            normalized_shape.push_back(1);
        }
        
        // Create weight and bias tensors with the same shape as normalized_shape
        torch::Tensor weight;
        torch::Tensor bias;
        
        // Decide whether to use weight and bias
        if (offset < Size) {
            bool use_weight_bias = Data[offset++] % 2 == 0;
            
            if (use_weight_bias) {
                // Create weight and bias with the same shape as normalized_shape
                weight = torch::ones(normalized_shape, input.options());
                bias = torch::zeros(normalized_shape, input.options());
            }
        }
        
        // Get eps value from the input data
        double eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure eps is positive and not too small
            if (std::isfinite(eps_raw)) {
                eps = std::abs(eps_raw);
                if (eps < 1e-10) eps = 1e-10;
                if (eps > 0.1) eps = 0.1;
            }
        }
        
        // Get elementwise_affine flag
        bool elementwise_affine = true;
        if (offset < Size) {
            elementwise_affine = Data[offset++] % 2 == 0;
        }
        
        // Apply layer_norm operation
        torch::Tensor output;
        
        if (weight.defined() && bias.defined()) {
            output = torch::layer_norm(input, normalized_shape, weight, bias, eps, elementwise_affine);
        } else {
            output = torch::layer_norm(input, normalized_shape, {}, {}, eps, elementwise_affine);
        }
        
        // Ensure the output is used to prevent optimization
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}