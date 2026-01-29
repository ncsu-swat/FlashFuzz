#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // layer_norm requires floating point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Extract normalized_shape from the input tensor
        std::vector<int64_t> normalized_shape;
        if (input.dim() > 0) {
            // Determine how many dimensions to normalize over (1 to min(3, dim))
            int64_t num_dims = 1;
            if (offset < Size) {
                num_dims = 1 + (Data[offset++] % std::min(static_cast<int64_t>(3), input.dim()));
            }
            
            // Use the last 'num_dims' dimensions as normalized_shape
            for (int64_t i = input.dim() - num_dims; i < input.dim(); i++) {
                normalized_shape.push_back(input.size(i));
            }
        } else {
            // For scalar tensors, use a default shape
            normalized_shape.push_back(1);
        }
        
        // Create weight and bias tensors with the same shape as normalized_shape
        torch::Tensor weight;
        torch::Tensor bias;
        
        // Decide whether to use weight and bias (elementwise_affine)
        bool use_weight_bias = false;
        if (offset < Size) {
            use_weight_bias = Data[offset++] % 2 == 0;
        }
        
        if (use_weight_bias) {
            // Create weight and bias with the same shape as normalized_shape
            weight = torch::ones(normalized_shape, input.options());
            bias = torch::zeros(normalized_shape, input.options());
            
            // Optionally randomize weight values for better coverage
            if (offset < Size && Data[offset++] % 2 == 0) {
                weight = torch::randn(normalized_shape, input.options());
            }
            if (offset < Size && Data[offset++] % 2 == 0) {
                bias = torch::randn(normalized_shape, input.options());
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
                eps = std::abs(static_cast<double>(eps_raw));
                if (eps < 1e-10) eps = 1e-10;
                if (eps > 0.1) eps = 0.1;
            }
        }
        
        // Apply layer_norm operation
        torch::Tensor output;
        
        try {
            if (weight.defined() && bias.defined()) {
                output = torch::layer_norm(input, normalized_shape, weight, bias, eps);
            } else {
                output = torch::layer_norm(input, normalized_shape, {}, {}, eps);
            }
        } catch (const c10::Error &e) {
            // Expected failures for invalid shapes/inputs - silently ignore
            return 0;
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