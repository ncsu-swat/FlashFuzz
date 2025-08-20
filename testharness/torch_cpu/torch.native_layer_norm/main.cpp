#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse normalized_shape from the remaining data
        std::vector<int64_t> normalized_shape;
        if (offset + 1 < Size) {
            uint8_t normalized_rank = Data[offset++] % 4; // Limit to reasonable rank
            normalized_shape = fuzzer_utils::parseShape(Data, offset, Size, normalized_rank);
        } else {
            // Default to normalizing the last dimension if no data left
            if (input.dim() > 0) {
                normalized_shape.push_back(input.size(-1));
            } else {
                normalized_shape.push_back(1);
            }
        }
        
        // Create weight and bias tensors with the same shape as normalized_shape
        torch::Tensor weight;
        torch::Tensor bias;
        
        // Create weight tensor
        if (offset < Size) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure weight has the same shape as normalized_shape
            if (weight.dim() > 0 && weight.sizes() != c10::IntArrayRef(normalized_shape)) {
                weight = weight.reshape(normalized_shape);
            }
        } else {
            weight = torch::ones(normalized_shape);
        }
        
        // Create bias tensor
        if (offset < Size) {
            bias = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure bias has the same shape as normalized_shape
            if (bias.dim() > 0 && bias.sizes() != c10::IntArrayRef(normalized_shape)) {
                bias = bias.reshape(normalized_shape);
            }
        } else {
            bias = torch::zeros(normalized_shape);
        }
        
        // Parse epsilon
        float eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure epsilon is positive and not too small
            if (eps <= 0 || std::isnan(eps)) {
                eps = 1e-5;
            }
        }
        
        // Call native_layer_norm
        auto output = at::native_layer_norm(
            input,
            normalized_shape,
            weight,
            bias,
            eps
        );
        
        // Unpack the output tuple (result, mean, rstd)
        auto& result = std::get<0>(output);
        auto& mean = std::get<1>(output);
        auto& rstd = std::get<2>(output);
        
        // Perform some operations on the results to ensure they're used
        auto sum = result.sum() + mean.sum() + rstd.sum();
        if (sum.item<float>() == -12345.6789f) {
            // This condition is unlikely to be true, just to prevent
            // the compiler from optimizing away the operations
            std::cerr << "Unexpected sum value" << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        return 0; // discard the input
    }
    return 0; // keep the input
}