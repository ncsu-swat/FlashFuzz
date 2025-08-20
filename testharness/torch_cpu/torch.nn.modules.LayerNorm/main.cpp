#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
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
        
        // Parse eps parameter
        double eps = 1e-5; // Default value
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure eps is positive and not too small
            if (std::isfinite(eps_raw) && eps_raw > 0) {
                eps = eps_raw;
            }
        }
        
        // Parse elementwise_affine parameter
        bool elementwise_affine = true; // Default value
        if (offset < Size) {
            elementwise_affine = Data[offset++] & 0x1;
        }
        
        // Create LayerNorm module
        torch::nn::LayerNorm layer_norm = torch::nn::LayerNorm(
            torch::nn::LayerNormOptions(normalized_shape)
                .eps(eps)
                .elementwise_affine(elementwise_affine)
        );
        
        // Apply LayerNorm to the input tensor
        torch::Tensor output = layer_norm->forward(input);
        
        // Try to access output properties to ensure computation completed
        auto output_size = output.sizes();
        auto output_dtype = output.dtype();
        
        // Try accessing parameters if they exist
        if (elementwise_affine) {
            auto weight = layer_norm->weight;
            auto bias = layer_norm->bias;
        }
        
        // Try different input types
        if (offset < Size) {
            try {
                auto dtype_selector = Data[offset++];
                auto new_dtype = fuzzer_utils::parseDataType(dtype_selector);
                auto input2 = input.to(new_dtype);
                auto output2 = layer_norm->forward(input2);
            } catch (const std::exception&) {
                // Ignore exceptions from type conversion
            }
        }
        
        // Try with different device if available
        if (torch::cuda::is_available() && offset < Size && (Data[offset++] & 0x1)) {
            try {
                auto input_cuda = input.cuda();
                layer_norm->to(torch::kCUDA);
                auto output_cuda = layer_norm->forward(input_cuda);
            } catch (const std::exception&) {
                // Ignore CUDA-related exceptions
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}