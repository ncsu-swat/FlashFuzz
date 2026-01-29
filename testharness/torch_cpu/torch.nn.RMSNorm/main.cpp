#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters for RMSNorm first
        // Get number of normalized dimensions (1-3)
        uint8_t num_norm_dims = (Data[offset++] % 3) + 1;
        
        // Get normalized_shape dimensions (must be >= 1)
        std::vector<int64_t> normalized_shape;
        for (uint8_t i = 0; i < num_norm_dims && offset < Size; i++) {
            int64_t dim_size = (Data[offset++] % 15) + 1; // 1-16
            normalized_shape.push_back(dim_size);
        }
        
        if (normalized_shape.empty()) {
            normalized_shape.push_back(4); // Default
        }
        
        // Build input tensor shape: add batch dimensions + normalized dimensions
        std::vector<int64_t> input_shape;
        
        // Add 1-2 batch dimensions
        if (offset < Size) {
            uint8_t num_batch_dims = (Data[offset++] % 2) + 1;
            for (uint8_t i = 0; i < num_batch_dims && offset < Size; i++) {
                int64_t batch_size = (Data[offset++] % 7) + 1; // 1-8
                input_shape.push_back(batch_size);
            }
        }
        if (input_shape.empty()) {
            input_shape.push_back(2); // Default batch dim
        }
        
        // Append normalized dimensions to input shape
        for (auto dim : normalized_shape) {
            input_shape.push_back(dim);
        }
        
        // Get epsilon parameter
        double epsilon = 1e-5;
        if (offset < Size) {
            uint8_t eps_idx = Data[offset++] % 5;
            double eps_values[] = {1e-8, 1e-6, 1e-5, 1e-4, 1e-3};
            epsilon = eps_values[eps_idx];
        }
        
        // Get options for weight
        bool use_weight = false;
        if (offset < Size) {
            use_weight = (Data[offset++] & 1);
        }
        
        // Get dtype
        torch::Dtype dtype = torch::kFloat32;
        if (offset < Size) {
            uint8_t dtype_idx = Data[offset++] % 3;
            torch::Dtype dtypes[] = {torch::kFloat32, torch::kFloat64, torch::kFloat16};
            dtype = dtypes[dtype_idx];
        }
        
        // Create input tensor with correct shape
        torch::Tensor input = torch::randn(input_shape, torch::TensorOptions().dtype(dtype));
        
        // Use remaining data to add variation to input
        if (offset < Size) {
            float scale = static_cast<float>(Data[offset++]) / 25.5f; // 0-10 range
            input = input * scale;
        }
        
        // Create weight tensor with shape matching normalized_shape
        std::optional<torch::Tensor> weight = std::nullopt;
        if (use_weight) {
            torch::Tensor weight_tensor = torch::ones(normalized_shape, torch::TensorOptions().dtype(dtype));
            // Add some variation to weights
            if (offset < Size) {
                float weight_scale = static_cast<float>(Data[offset++]) / 127.5f; // 0-2 range
                weight_tensor = weight_tensor * weight_scale;
            }
            weight = weight_tensor;
        }
        
        // Apply RMSNorm using torch::rms_norm (not torch::nn::functional::rms_norm)
        torch::Tensor output = torch::rms_norm(
            input, 
            normalized_shape, 
            weight,
            epsilon
        );
        
        // Verify output shape matches input shape
        if (output.sizes() != input.sizes()) {
            std::cerr << "Shape mismatch!" << std::endl;
        }
        
        // Force computation
        volatile float result = output.mean().item<float>();
        (void)result;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}