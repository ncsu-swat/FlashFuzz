#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has 5 dimensions (N, C, D, H, W) for BNReLU3d
        if (input.dim() != 5) {
            // Reshape to 5D if needed
            int64_t total_elements = input.numel();
            int64_t batch_size = 1;
            int64_t channels = 1;
            int64_t depth = 1;
            int64_t height = 1;
            int64_t width = 1;
            
            if (total_elements > 0) {
                // Distribute elements across dimensions
                if (offset + 5 <= Size) {
                    batch_size = (Data[offset++] % 4) + 1;
                    channels = (Data[offset++] % 4) + 1;
                    depth = (Data[offset++] % 4) + 1;
                    height = (Data[offset++] % 4) + 1;
                    width = total_elements / (batch_size * channels * depth * height);
                    if (width <= 0) width = 1;
                }
                
                // Ensure we have at least one element per dimension
                int64_t new_total = batch_size * channels * depth * height * width;
                if (new_total > total_elements) {
                    // Adjust dimensions to fit
                    width = 1;
                    height = 1;
                    depth = 1;
                    channels = 1;
                    batch_size = total_elements;
                }
            }
            
            // Reshape tensor
            input = input.reshape({batch_size, channels, depth, height, width});
        }
        
        // Ensure input is quantized
        if (!input.is_quantized()) {
            // Get scale and zero_point from the fuzzer data
            double scale = 0.1;
            int64_t zero_point = 0;
            
            if (offset + sizeof(double) + sizeof(int64_t) <= Size) {
                memcpy(&scale, Data + offset, sizeof(double));
                offset += sizeof(double);
                memcpy(&zero_point, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            }
            
            // Ensure scale is positive and reasonable
            scale = std::abs(scale);
            if (scale < 1e-10) scale = 0.1;
            if (scale > 1e10) scale = 0.1;
            
            // Ensure zero_point is in valid range for quint8
            zero_point = zero_point % 256;
            if (zero_point < 0) zero_point += 256;
            
            // Quantize the tensor
            input = torch::quantize_per_tensor(input, scale, zero_point, torch::kQUInt8);
        }
        
        // Get parameters for BNReLU3d
        int64_t num_features = input.size(1); // Number of channels
        
        // Create parameters for BNReLU3d
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + sizeof(double) * 2 <= Size) {
            memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Ensure eps is positive and reasonable
        eps = std::abs(eps);
        if (eps < 1e-10) eps = 1e-5;
        if (eps > 1.0) eps = 1e-5;
        
        // Ensure momentum is between 0 and 1
        momentum = std::abs(momentum);
        if (momentum > 1.0) momentum = 0.1;
        
        // Create running_mean and running_var tensors
        auto running_mean = torch::zeros({num_features});
        auto running_var = torch::ones({num_features});
        
        // Create weight and bias tensors
        auto weight = torch::ones({num_features});
        auto bias = torch::zeros({num_features});
        
        // Create BatchNorm3d module
        auto bn = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(num_features)
                                         .eps(eps)
                                         .momentum(momentum)
                                         .affine(true)
                                         .track_running_stats(true));
        
        // Set the parameters
        bn->weight = weight;
        bn->bias = bias;
        bn->running_mean = running_mean;
        bn->running_var = running_var;
        
        // Apply batch normalization followed by ReLU manually
        auto bn_output = bn->forward(input);
        auto output = torch::relu(bn_output);
        
        // Access some properties of the output to ensure it's used
        auto sizes = output.sizes();
        auto numel = output.numel();
        auto dtype = output.dtype();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
