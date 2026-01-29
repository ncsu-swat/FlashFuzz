#include "fuzzer_utils.h"
#include <iostream>
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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create ReLU module with different configurations
        bool inplace = false;
        if (offset < Size) {
            inplace = Data[offset++] & 0x1;
        }
        
        // Create ReLU module
        torch::nn::ReLU relu_module(torch::nn::ReLUOptions().inplace(inplace));
        
        // Apply ReLU operation
        // For inplace, we need to clone the input first
        torch::Tensor relu_input = inplace ? input.clone() : input;
        torch::Tensor output = relu_module->forward(relu_input);
        
        // Test functional version as well
        torch::Tensor output_functional = torch::relu(input);
        
        // Test with threshold parameter (ReLU6-like behavior using clamp)
        double threshold = 6.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&threshold, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize threshold value
            if (!std::isfinite(threshold)) {
                threshold = 6.0;
            }
            threshold = std::abs(threshold);
            if (threshold > 1e6) {
                threshold = 1e6;
            }
        }
        
        // Apply threshold clamp (ReLU6-like)
        try {
            torch::Tensor output_threshold = torch::clamp(input, 0.0, threshold);
        } catch (...) {
            // Clamp may fail for certain dtypes, ignore silently
        }
        
        // Test with leaky ReLU parameters
        double negative_slope = 0.01;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&negative_slope, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize negative_slope value
            if (!std::isfinite(negative_slope)) {
                negative_slope = 0.01;
            }
            // Clamp to reasonable range
            negative_slope = std::max(-1.0, std::min(1.0, negative_slope));
        }
        
        // Apply leaky ReLU
        torch::Tensor output_leaky = torch::leaky_relu(input, negative_slope);
        
        // Test edge cases with different tensor types
        if (offset < Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor relu_input2 = inplace ? input2.clone() : input2;
            torch::Tensor output2 = relu_module->forward(relu_input2);
        }
        
        // Test with zero-sized dimensions
        try {
            std::vector<int64_t> empty_shape = {0, 2, 3};
            torch::Tensor empty_tensor = torch::empty(empty_shape);
            torch::Tensor empty_output = relu_module->forward(empty_tensor);
        } catch (...) {
            // Empty tensor handling may vary
        }
        
        // Test with scalar tensor
        try {
            torch::Tensor scalar_tensor = torch::tensor(-5.0);
            torch::Tensor scalar_output = relu_module->forward(scalar_tensor);
        } catch (...) {
            // Scalar handling may vary
        }
        
        // Test ReLU6 module explicitly
        try {
            torch::nn::ReLU6 relu6_module(torch::nn::ReLU6Options().inplace(false));
            torch::Tensor output_relu6 = relu6_module->forward(input);
        } catch (...) {
            // ReLU6 may not support all dtypes
        }
        
        // Test PReLU (parametric ReLU) with learnable parameter
        try {
            torch::nn::PReLU prelu_module(torch::nn::PReLUOptions().num_parameters(1));
            torch::Tensor float_input = input.to(torch::kFloat32);
            torch::Tensor output_prelu = prelu_module->forward(float_input);
        } catch (...) {
            // PReLU requires float tensors
        }
        
        // Test with different tensor shapes
        if (offset + 2 < Size) {
            int64_t batch_size = (Data[offset++] % 8) + 1;
            int64_t features = (Data[offset++] % 16) + 1;
            
            try {
                torch::Tensor shaped_input = torch::randn({batch_size, features});
                torch::Tensor shaped_output = relu_module->forward(shaped_input);
            } catch (...) {
                // Shape handling
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