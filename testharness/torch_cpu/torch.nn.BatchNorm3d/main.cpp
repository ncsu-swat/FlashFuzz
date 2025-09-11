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
        if (Size < 5) {
            return 0;
        }
        
        // Create input tensor - must be 5D for BatchNorm3d (N, C, D, H, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 1 more byte for parameters
        if (offset >= Size) {
            return 0;
        }
        
        // Extract parameters for BatchNorm3d
        uint8_t param_byte = Data[offset++];
        
        // Get number of features (channels) from input if possible
        int64_t num_features = 1;
        if (input.dim() >= 2) {
            num_features = input.size(1);
        }
        
        // If num_features is 0, set it to 1 to avoid invalid argument
        if (num_features <= 0) {
            num_features = 1;
        }
        
        // Parse other parameters from the fuzzer data
        bool affine = (param_byte & 0x01) != 0;
        bool track_running_stats = (param_byte & 0x02) != 0;
        double momentum = (param_byte & 0x04) ? 0.1 : 0.01;
        double eps = (param_byte & 0x08) ? 1e-5 : 1e-4;
        
        // Create BatchNorm3d module
        torch::nn::BatchNorm3d bn(torch::nn::BatchNorm3dOptions(num_features)
                                  .eps(eps)
                                  .momentum(momentum)
                                  .affine(affine)
                                  .track_running_stats(track_running_stats));
        
        // If input doesn't have 5 dimensions, reshape it to make it compatible
        if (input.dim() != 5) {
            std::vector<int64_t> new_shape;
            
            // Batch size
            new_shape.push_back(1);
            
            // Channels (num_features)
            new_shape.push_back(num_features);
            
            // Depth, Height, Width
            for (int i = 0; i < 3; i++) {
                if (i < input.dim() - 1) {
                    new_shape.push_back(input.size(i + 1));
                } else {
                    new_shape.push_back(1);
                }
            }
            
            // Reshape the tensor to 5D
            input = input.reshape(new_shape);
        }
        
        // Apply BatchNorm3d
        torch::Tensor output = bn->forward(input);
        
        // Test training mode
        bn->train();
        torch::Tensor output_train = bn->forward(input);
        
        // Test eval mode
        bn->eval();
        torch::Tensor output_eval = bn->forward(input);
        
        // Access running mean and variance
        if (track_running_stats) {
            auto running_mean = bn->running_mean;
            auto running_var = bn->running_var;
        }
        
        // Access weight and bias if affine is true
        if (affine) {
            auto weight = bn->weight;
            auto bias = bn->bias;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
