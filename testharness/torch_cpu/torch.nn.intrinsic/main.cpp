#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight tensor
        torch::Tensor weight;
        if (offset < Size) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            weight = torch::randn({3, 3, 3, 3});
        }
        
        // Create bias tensor (optional)
        torch::Tensor bias;
        bool use_bias = offset < Size && Data[offset++] % 2 == 0;
        if (use_bias && offset < Size) {
            bias = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Test basic conv operations since intrinsic modules are not available
        
        // 1. Conv2d + BatchNorm2d + ReLU sequence
        try {
            int64_t num_features = weight.size(0);
            auto conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(input.size(1), num_features, {3, 3}).padding(1)
            );
            auto bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_features));
            auto relu = torch::nn::ReLU();
            
            // Set weights
            conv->weight = weight;
            if (use_bias) {
                conv->bias = bias;
            }
            
            // Forward pass
            auto conv_out = conv->forward(input);
            auto bn_out = bn->forward(conv_out);
            auto output = relu->forward(bn_out);
        } catch (...) {
            // Ignore exceptions from this test
        }
        
        // 2. Conv2d + ReLU sequence
        try {
            int64_t num_features = weight.size(0);
            auto conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(input.size(1), num_features, {3, 3}).padding(1)
            );
            auto relu = torch::nn::ReLU();
            
            // Set weights
            conv->weight = weight;
            if (use_bias) {
                conv->bias = bias;
            }
            
            // Forward pass
            auto conv_out = conv->forward(input);
            auto output = relu->forward(conv_out);
        } catch (...) {
            // Ignore exceptions from this test
        }
        
        // 3. Linear + ReLU sequence
        try {
            auto linear = torch::nn::Linear(torch::nn::LinearOptions(input.size(-1), 10));
            auto relu = torch::nn::ReLU();
            
            // Forward pass
            auto linear_out = linear->forward(input);
            auto output = relu->forward(linear_out);
        } catch (...) {
            // Ignore exceptions from this test
        }
        
        // 4. BatchNorm2d + ReLU sequence
        try {
            int64_t num_features = input.size(1);
            auto bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_features));
            auto relu = torch::nn::ReLU();
            
            // Forward pass
            auto bn_out = bn->forward(input);
            auto output = relu->forward(bn_out);
        } catch (...) {
            // Ignore exceptions from this test
        }
        
        // 5. Conv1d + BatchNorm1d sequence
        try {
            int64_t num_features = weight.size(0);
            auto conv = torch::nn::Conv1d(
                torch::nn::Conv1dOptions(input.size(1), num_features, 3).padding(1)
            );
            auto bn = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(num_features));
            
            // Forward pass
            auto conv_out = conv->forward(input);
            auto output = bn->forward(conv_out);
        } catch (...) {
            // Ignore exceptions from this test
        }
        
        // 6. Conv1d + ReLU sequence
        try {
            int64_t num_features = weight.size(0);
            auto conv = torch::nn::Conv1d(
                torch::nn::Conv1dOptions(input.size(1), num_features, 3).padding(1)
            );
            auto relu = torch::nn::ReLU();
            
            // Forward pass
            auto conv_out = conv->forward(input);
            auto output = relu->forward(conv_out);
        } catch (...) {
            // Ignore exceptions from this test
        }
        
        // 7. Conv3d + ReLU sequence
        try {
            int64_t num_features = weight.size(0);
            auto conv = torch::nn::Conv3d(
                torch::nn::Conv3dOptions(input.size(1), num_features, 3).padding(1)
            );
            auto relu = torch::nn::ReLU();
            
            // Forward pass
            auto conv_out = conv->forward(input);
            auto output = relu->forward(conv_out);
        } catch (...) {
            // Ignore exceptions from this test
        }
        
        // 8. BatchNorm3d + ReLU sequence
        try {
            int64_t num_features = input.size(1);
            auto bn = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(num_features));
            auto relu = torch::nn::ReLU();
            
            // Forward pass
            auto bn_out = bn->forward(input);
            auto output = relu->forward(bn_out);
        } catch (...) {
            // Ignore exceptions from this test
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}