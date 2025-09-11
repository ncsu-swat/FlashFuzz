#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor - must be 5D for BatchNorm3d
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the input tensor is 5D (N, C, D, H, W)
        if (input.dim() != 5) {
            // Reshape to 5D if needed
            int64_t total_elements = input.numel();
            
            // Extract parameters for dimensions from the data
            int64_t N = 1, C = 1, D = 1, H = 1, W = 1;
            
            if (offset + 5 <= Size) {
                N = (Data[offset++] % 4) + 1;  // 1-4 batch size
                C = (Data[offset++] % 4) + 1;  // 1-4 channels
                D = (Data[offset++] % 4) + 1;  // 1-4 depth
                H = (Data[offset++] % 4) + 1;  // 1-4 height
                W = (Data[offset++] % 4) + 1;  // 1-4 width
            }
            
            // Calculate total elements needed
            int64_t needed_elements = N * C * D * H * W;
            
            // If we don't have enough elements, adjust dimensions
            if (total_elements < needed_elements) {
                // Simple approach: set all dimensions to 1 except the last one
                N = C = D = H = 1;
                W = total_elements > 0 ? total_elements : 1;
            } else if (total_elements > needed_elements) {
                // Adjust the last dimension to use all elements
                W = (total_elements / (N * C * D * H));
            }
            
            // Reshape the tensor
            input = input.reshape({N, C, D, H, W});
        }
        
        // Ensure the input tensor has float type for quantized operations
        if (input.scalar_type() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Extract parameters for BatchNorm3d
        int64_t num_features = input.size(1); // Number of channels
        
        // Extract additional parameters from the data
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        if (offset + 4 <= Size) {
            // Extract eps (small value to avoid division by zero)
            uint8_t eps_byte = Data[offset++];
            eps = 1e-5 + (eps_byte % 10) * 1e-5;
            
            // Extract momentum
            uint8_t momentum_byte = Data[offset++];
            momentum = 0.1 + (momentum_byte % 9) * 0.1;
            
            // Extract boolean parameters
            affine = (Data[offset++] % 2) == 1;
            track_running_stats = (Data[offset++] % 2) == 1;
        }
        
        // Create regular BatchNorm3d module (quantized version not available in C++ frontend)
        torch::nn::BatchNorm3dOptions options(num_features);
        options.eps(eps).momentum(momentum).affine(affine).track_running_stats(track_running_stats);
        
        torch::nn::BatchNorm3d bn(options);
        
        // If affine is true, initialize weight and bias
        if (affine) {
            auto weight = torch::ones({num_features});
            auto bias = torch::zeros({num_features});
            
            // Modify weight and bias with data if available
            if (offset + 2 * num_features <= Size) {
                for (int64_t i = 0; i < num_features; i++) {
                    if (offset < Size) {
                        weight[i] = static_cast<float>(Data[offset++]) / 255.0f;
                    }
                }
                for (int64_t i = 0; i < num_features; i++) {
                    if (offset < Size) {
                        bias[i] = static_cast<float>(Data[offset++]) / 255.0f - 0.5f;
                    }
                }
            }
            
            bn->weight = weight;
            bn->bias = bias;
        }
        
        // Forward pass
        auto output = bn->forward(input);
        
        // Try to access some properties to ensure they're computed
        if (track_running_stats) {
            auto running_mean = bn->running_mean;
            auto running_var = bn->running_var;
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
