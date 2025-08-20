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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the tensor has 5 dimensions for InstanceNorm3d (N, C, D, H, W)
        if (input.dim() != 5) {
            // Reshape to 5D if needed
            int64_t total_elements = input.numel();
            
            // Extract parameters from remaining data
            uint8_t n_val = (offset < Size) ? Data[offset++] : 1;
            uint8_t c_val = (offset < Size) ? Data[offset++] : 1;
            uint8_t d_val = (offset < Size) ? Data[offset++] : 1;
            
            // Ensure values are at least 1
            int64_t n = (n_val % 4) + 1;
            int64_t c = (c_val % 4) + 1;
            int64_t d = (d_val % 4) + 1;
            
            // Calculate h and w to maintain total elements
            int64_t remaining = total_elements / (n * c * d);
            int64_t h = std::max<int64_t>(1, static_cast<int64_t>(std::sqrt(remaining)));
            int64_t w = std::max<int64_t>(1, remaining / h);
            
            // Adjust dimensions to ensure total elements match
            while (n * c * d * h * w > total_elements) {
                if (w > 1) w--;
                else if (h > 1) h--;
                else if (d > 1) d--;
                else if (c > 1) c--;
                else if (n > 1) n--;
                else break;
            }
            
            // Create a new tensor with the desired shape
            if (n * c * d * h * w > 0) {
                try {
                    input = input.reshape({n, c, d, h, w});
                } catch (...) {
                    // If reshape fails, create a new tensor
                    input = torch::ones({n, c, d, h, w}, input.options());
                }
            } else {
                // Fallback to a minimal valid shape
                input = torch::ones({1, 1, 1, 1, 1}, input.options());
            }
        }
        
        // Ensure input is quantized
        if (!input.is_quantized()) {
            // Quantize the tensor
            auto scale = 1.0f / 256.0f;
            auto zero_point = 0;
            
            if (offset + 4 <= Size) {
                // Extract scale from data
                float extracted_scale;
                std::memcpy(&extracted_scale, Data + offset, sizeof(float));
                offset += sizeof(float);
                
                // Ensure scale is positive and reasonable
                if (extracted_scale > 0 && extracted_scale < 100.0f) {
                    scale = extracted_scale;
                }
            }
            
            if (offset < Size) {
                // Extract zero_point from data
                zero_point = static_cast<int>(Data[offset++]);
            }
            
            // Quantize the tensor
            input = torch::quantize_per_tensor(input, scale, zero_point, torch::kQUInt8);
        }
        
        // Extract parameters for InstanceNorm3d
        int64_t num_features = input.size(1); // Number of channels
        
        // Extract eps from data
        double eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            float extracted_eps;
            std::memcpy(&extracted_eps, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure eps is positive
            if (extracted_eps > 0 && extracted_eps < 1.0f) {
                eps = extracted_eps;
            }
        }
        
        // Extract momentum from data
        double momentum = 0.1;
        if (offset + sizeof(float) <= Size) {
            float extracted_momentum;
            std::memcpy(&extracted_momentum, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure momentum is in valid range
            if (extracted_momentum >= 0 && extracted_momentum <= 1.0f) {
                momentum = extracted_momentum;
            }
        }
        
        // Extract affine flag
        bool affine = false;
        if (offset < Size) {
            affine = (Data[offset++] % 2) == 1;
        }
        
        // Extract track_running_stats flag
        bool track_running_stats = false;
        if (offset < Size) {
            track_running_stats = (Data[offset++] % 2) == 1;
        }
        
        // Create weight and bias tensors if affine is true
        torch::Tensor weight = torch::Tensor();
        torch::Tensor bias = torch::Tensor();
        
        if (affine) {
            weight = torch::ones({num_features});
            bias = torch::zeros({num_features});
        }
        
        // Create running mean and var tensors if track_running_stats is true
        torch::Tensor running_mean = torch::Tensor();
        torch::Tensor running_var = torch::Tensor();
        
        if (track_running_stats) {
            running_mean = torch::zeros({num_features});
            running_var = torch::ones({num_features});
        }
        
        // Apply instance normalization using the functional API
        torch::Tensor output = torch::instance_norm(
            input,
            affine ? c10::optional<torch::Tensor>(weight) : c10::nullopt,
            affine ? c10::optional<torch::Tensor>(bias) : c10::nullopt,
            track_running_stats ? c10::optional<torch::Tensor>(running_mean) : c10::nullopt,
            track_running_stats ? c10::optional<torch::Tensor>(running_var) : c10::nullopt,
            true, // use_input_stats
            momentum,
            eps,
            false // cudnn_enabled
        );
        
        // Use the output to prevent optimization
        auto sum = output.sum();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}