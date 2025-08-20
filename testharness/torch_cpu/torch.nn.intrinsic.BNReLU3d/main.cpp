#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor for BNReLU3d
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the input tensor is 5D (batch_size, channels, depth, height, width)
        // If not, reshape it to a valid 5D shape
        if (input.dim() != 5) {
            std::vector<int64_t> new_shape;
            if (input.dim() > 5) {
                // If tensor has more than 5 dimensions, collapse extra dimensions
                new_shape = {input.size(0), input.size(1), input.size(2), input.size(3), 1};
                for (int i = 4; i < input.dim(); i++) {
                    new_shape[4] *= input.size(i);
                }
            } else {
                // If tensor has fewer than 5 dimensions, add dimensions
                new_shape = {1, 1, 1, 1, 1};
                for (int i = 0; i < input.dim(); i++) {
                    new_shape[i] = input.size(i);
                }
            }
            
            // Reshape the tensor
            input = input.reshape(new_shape);
        }
        
        // Get number of features (channels)
        int64_t num_features = input.size(1);
        
        // Create parameters for BNReLU3d
        auto weight = torch::ones({num_features});
        auto bias = torch::zeros({num_features});
        auto running_mean = torch::zeros({num_features});
        auto running_var = torch::ones({num_features});
        
        // Create BNReLU3d module
        auto bn = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(num_features));
        
        // Set parameters
        bn->weight = weight;
        bn->bias = bias;
        bn->running_mean = running_mean;
        bn->running_var = running_var;
        
        // Create ReLU module
        auto relu = torch::nn::ReLU();
        
        // Apply BNReLU3d (sequential application of BatchNorm3d and ReLU)
        torch::Tensor output;
        
        // First apply BatchNorm3d
        output = bn->forward(input);
        
        // Then apply ReLU
        output = relu->forward(output);
        
        // Alternatively, we can use a sequential module
        auto bnrelu3d = torch::nn::Sequential(
            bn,
            relu
        );
        
        // Apply the sequential module
        torch::Tensor output2 = bnrelu3d->forward(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        auto mean = output.mean();
        auto max_val = output.max();
        
        // Try different training modes
        bn->eval();
        torch::Tensor eval_output = bn->forward(input);
        eval_output = relu->forward(eval_output);
        
        bn->train();
        torch::Tensor train_output = bn->forward(input);
        train_output = relu->forward(train_output);
        
        // Try with different eps values
        if (offset < Size) {
            double eps = static_cast<double>(Data[offset++]) / 255.0;
            bn->options.eps(eps);
            torch::Tensor eps_output = bn->forward(input);
            eps_output = relu->forward(eps_output);
        }
        
        // Try with different momentum values
        if (offset < Size) {
            double momentum = static_cast<double>(Data[offset++]) / 255.0;
            bn->options.momentum(momentum);
            torch::Tensor momentum_output = bn->forward(input);
            momentum_output = relu->forward(momentum_output);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}