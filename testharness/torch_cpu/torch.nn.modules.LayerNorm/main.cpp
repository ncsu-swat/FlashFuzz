#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is floating point for LayerNorm
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Get normalized_shape from the input tensor
        std::vector<int64_t> normalized_shape;
        if (input.dim() > 0) {
            // Take the last dimension(s) as normalized_shape
            uint8_t num_normalized_dims = 1;
            if (offset < Size) {
                num_normalized_dims = (Data[offset++] % input.dim()) + 1;
            }
            
            for (int64_t i = input.dim() - num_normalized_dims; i < input.dim(); i++) {
                normalized_shape.push_back(input.size(i));
            }
        } else {
            // For scalar tensors, use a default shape
            normalized_shape.push_back(1);
        }
        
        // Validate normalized_shape is not empty
        if (normalized_shape.empty()) {
            return 0;
        }
        
        // Parse eps parameter
        double eps = 1e-5; // Default value
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure eps is positive and reasonable
            if (std::isfinite(eps_raw) && eps_raw > 1e-12 && eps_raw < 1.0) {
                eps = static_cast<double>(eps_raw);
            }
        }
        
        // Parse elementwise_affine parameter
        bool elementwise_affine = true; // Default value
        if (offset < Size) {
            elementwise_affine = Data[offset++] & 0x1;
        }
        
        // Create LayerNorm module
        torch::nn::LayerNorm layer_norm(
            torch::nn::LayerNormOptions(normalized_shape)
                .eps(eps)
                .elementwise_affine(elementwise_affine)
        );
        
        // Apply LayerNorm to the input tensor
        torch::Tensor output = layer_norm->forward(input);
        
        // Verify output properties
        auto output_size = output.sizes();
        auto output_dtype = output.dtype();
        (void)output_size;
        (void)output_dtype;
        
        // Access parameters if they exist
        if (elementwise_affine) {
            auto weight = layer_norm->weight;
            auto bias = layer_norm->bias;
            (void)weight;
            (void)bias;
        }
        
        // Test with different input that matches normalized_shape
        if (offset < Size) {
            try {
                // Create a new input with shape that ends with normalized_shape
                std::vector<int64_t> new_shape;
                uint8_t batch_dims = (Data[offset++] % 3) + 1; // 1-3 batch dims
                for (uint8_t i = 0; i < batch_dims && offset < Size; i++) {
                    int64_t dim_size = (Data[offset++] % 8) + 1;
                    new_shape.push_back(dim_size);
                }
                for (auto dim : normalized_shape) {
                    new_shape.push_back(dim);
                }
                
                auto input2 = torch::randn(new_shape);
                auto output2 = layer_norm->forward(input2);
                (void)output2;
            } catch (const std::exception&) {
                // Ignore exceptions from shape mismatches
            }
        }
        
        // Test eval mode
        if (offset < Size && (Data[offset++] & 0x1)) {
            layer_norm->eval();
            auto output_eval = layer_norm->forward(input);
            (void)output_eval;
        }
        
        // Test with different floating point types
        if (offset < Size) {
            try {
                uint8_t dtype_sel = Data[offset++] % 3;
                torch::Dtype new_dtype;
                switch (dtype_sel) {
                    case 0: new_dtype = torch::kFloat32; break;
                    case 1: new_dtype = torch::kFloat64; break;
                    default: new_dtype = torch::kFloat32; break;
                }
                
                auto input_typed = input.to(new_dtype);
                
                // Create new LayerNorm for the new dtype
                torch::nn::LayerNorm layer_norm_typed(
                    torch::nn::LayerNormOptions(normalized_shape)
                        .eps(eps)
                        .elementwise_affine(elementwise_affine)
                );
                layer_norm_typed->to(new_dtype);
                
                auto output_typed = layer_norm_typed->forward(input_typed);
                (void)output_typed;
            } catch (const std::exception&) {
                // Ignore dtype conversion exceptions
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