#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // LayerNorm requires at least 1D tensor
        if (input.dim() == 0) {
            input = input.unsqueeze(0);
        }
        
        // Ensure input is float type (LayerNorm requires floating point)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Get normalized_shape from the input tensor
        std::vector<int64_t> normalized_shape;
        
        // Take the last dimension(s) as normalized_shape
        uint8_t num_normalized_dims = 1;
        if (offset < Size) {
            num_normalized_dims = (Data[offset++] % input.dim()) + 1;
        }
        
        for (int64_t i = input.dim() - num_normalized_dims; i < input.dim(); i++) {
            normalized_shape.push_back(input.size(i));
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
        
        // Test with different dtypes using same normalized_shape
        if (offset < Size && (Data[offset] & 0x1)) {
            try {
                auto input_f64 = input.to(torch::kFloat64);
                
                // Create new LayerNorm for f64
                torch::nn::LayerNorm layer_norm_f64(
                    torch::nn::LayerNormOptions(normalized_shape)
                        .eps(eps)
                        .elementwise_affine(elementwise_affine)
                );
                layer_norm_f64->to(torch::kFloat64);
                
                auto output_f64 = layer_norm_f64->forward(input_f64);
                (void)output_f64;
            } catch (...) {
                // Ignore expected failures from type conversion
            }
        }
        
        // Test eval mode vs train mode
        if (offset + 1 < Size && (Data[offset + 1] & 0x1)) {
            layer_norm->eval();
            torch::Tensor output_eval = layer_norm->forward(input);
            (void)output_eval;
            
            layer_norm->train();
            torch::Tensor output_train = layer_norm->forward(input);
            (void)output_train;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}