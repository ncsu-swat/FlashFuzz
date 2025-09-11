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
        
        // Create input tensor - must be 5D for BatchNorm3d (N, C, D, H, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the tensor has 5 dimensions for BatchNorm3d
        if (input.dim() != 5) {
            // Reshape to 5D if needed
            std::vector<int64_t> new_shape;
            if (input.dim() < 5) {
                // Add dimensions if needed
                new_shape = input.sizes().vec();
                while (new_shape.size() < 5) {
                    new_shape.push_back(1);
                }
            } else if (input.dim() > 5) {
                // Collapse extra dimensions
                new_shape.push_back(input.size(0)); // N
                new_shape.push_back(input.size(1)); // C
                
                // Collapse remaining dimensions into D, H, W
                int64_t d_size = 1;
                for (int i = 2; i < input.dim() - 2; i++) {
                    d_size *= input.size(i);
                }
                new_shape.push_back(d_size); // D
                
                new_shape.push_back(input.size(input.dim() - 2)); // H
                new_shape.push_back(input.size(input.dim() - 1)); // W
            }
            
            if (!new_shape.empty()) {
                input = input.reshape(new_shape);
            }
        }
        
        // Ensure we have at least one channel dimension
        if (input.size(1) == 0) {
            return 0;
        }
        
        // Extract parameters for BatchNorm3d from the remaining data
        bool affine = true;
        bool track_running_stats = true;
        double momentum = 0.1;
        double eps = 1e-5;
        
        if (offset + 4 <= Size) {
            // Use some bytes to determine parameters
            affine = Data[offset++] & 0x1;
            track_running_stats = Data[offset++] & 0x1;
            
            // Parse momentum (between 0 and 1)
            if (offset < Size) {
                momentum = static_cast<double>(Data[offset++]) / 255.0;
            }
            
            // Parse epsilon (small positive value)
            if (offset < Size) {
                eps = std::max(1e-10, static_cast<double>(Data[offset++]) / 1000.0);
            }
        }
        
        // Get number of features (channels) from the input tensor
        int64_t num_features = input.size(1);
        
        // Create BatchNorm3d module
        torch::nn::BatchNorm3d bn(torch::nn::BatchNorm3dOptions(num_features)
                                  .eps(eps)
                                  .momentum(momentum)
                                  .affine(affine)
                                  .track_running_stats(track_running_stats));
        
        // Apply BatchNorm3d to the input tensor
        torch::Tensor output = bn->forward(input);
        
        // Try different modes (training vs evaluation)
        if (offset < Size && (Data[offset++] & 0x1)) {
            bn->eval();
            torch::Tensor eval_output = bn->forward(input);
        }
        
        // Try with different data types if possible
        if (input.dtype() != torch::kFloat && input.dtype() != torch::kDouble) {
            // Convert to float for numerical stability
            torch::Tensor float_input = input.to(torch::kFloat);
            torch::Tensor float_output = bn->forward(float_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
