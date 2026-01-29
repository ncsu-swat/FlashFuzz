#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        
        if (Size < 10) {
            return -1;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 1 dimension
        if (input.dim() == 0) {
            input = input.unsqueeze(0);
        }
        
        // Parse normalized_shape from the remaining data
        // normalized_shape must match the trailing dimensions of input
        std::vector<int64_t> normalized_shape;
        
        if (offset + 1 < Size) {
            // Determine how many trailing dimensions to normalize (1 to input.dim())
            uint8_t num_dims = (Data[offset++] % input.dim()) + 1;
            
            // Get the trailing dimensions of input
            for (int i = input.dim() - num_dims; i < input.dim(); i++) {
                normalized_shape.push_back(input.size(i));
            }
        } else {
            // Default: normalize the last dimension
            normalized_shape.push_back(input.size(-1));
        }
        
        // Ensure normalized_shape is not empty and has valid sizes
        if (normalized_shape.empty()) {
            normalized_shape.push_back(input.size(-1));
        }
        
        // Create weight and bias with correct shape (matching normalized_shape)
        torch::Tensor weight;
        torch::Tensor bias;
        
        // Decide whether to use weight/bias based on fuzzer data
        bool use_weight = (offset < Size) ? (Data[offset++] % 2 == 1) : true;
        bool use_bias = (offset < Size) ? (Data[offset++] % 2 == 1) : true;
        
        if (use_weight) {
            // Create weight tensor with shape matching normalized_shape
            weight = torch::randn(normalized_shape, input.options());
            // Optionally modify with fuzzer data
            if (offset + sizeof(float) <= Size) {
                float scale;
                std::memcpy(&scale, Data + offset, sizeof(float));
                offset += sizeof(float);
                if (std::isfinite(scale) && scale != 0.0f) {
                    weight = weight * scale;
                }
            }
        }
        
        if (use_bias) {
            // Create bias tensor with shape matching normalized_shape
            bias = torch::randn(normalized_shape, input.options());
            if (offset + sizeof(float) <= Size) {
                float scale;
                std::memcpy(&scale, Data + offset, sizeof(float));
                offset += sizeof(float);
                if (std::isfinite(scale)) {
                    bias = bias * scale;
                }
            }
        }
        
        // Parse epsilon
        double eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            float eps_f;
            std::memcpy(&eps_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure epsilon is positive and reasonable
            if (std::isfinite(eps_f) && eps_f > 1e-12 && eps_f < 1.0) {
                eps = static_cast<double>(eps_f);
            }
        }
        
        // Call native_layer_norm
        // The function signature: native_layer_norm(input, normalized_shape, weight, bias, eps)
        // weight and bias can be undefined tensors
        auto output = torch::native_layer_norm(
            input,
            normalized_shape,
            use_weight ? weight : torch::Tensor(),
            use_bias ? bias : torch::Tensor(),
            eps
        );
        
        // Unpack the output tuple (result, mean, rstd)
        auto& result = std::get<0>(output);
        auto& mean = std::get<1>(output);
        auto& rstd = std::get<2>(output);
        
        // Verify outputs are valid
        if (result.numel() > 0) {
            auto sum = result.sum();
            (void)sum;
        }
        if (mean.numel() > 0) {
            auto mean_sum = mean.sum();
            (void)mean_sum;
        }
        if (rstd.numel() > 0) {
            auto rstd_sum = rstd.sum();
            (void)rstd_sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}