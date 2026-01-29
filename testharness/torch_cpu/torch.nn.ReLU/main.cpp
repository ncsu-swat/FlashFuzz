#include "fuzzer_utils.h"
#include <iostream>
#include <limits>

// --- Fuzzer Entry Point ---
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
        
        // Skip if there's not enough data
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
        torch::nn::ReLU relu(torch::nn::ReLUOptions().inplace(inplace));
        
        // Apply ReLU operation - clone if inplace to avoid modifying original
        if (inplace) {
            torch::Tensor input_copy = input.clone();
            torch::Tensor output = relu->forward(input_copy);
        } else {
            torch::Tensor output = relu->forward(input);
        }
        
        // Try different variants of ReLU
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 3;
            
            if (variant == 0) {
                // Test functional interface
                torch::Tensor output2 = torch::relu(input);
            } else if (variant == 1) {
                // Test inplace functional interface
                torch::Tensor input_copy = input.clone();
                torch::relu_(input_copy);
            } else {
                // Test with ReLU6
                torch::nn::ReLU6 relu6(torch::nn::ReLU6Options().inplace(false));
                torch::Tensor output3 = relu6->forward(input);
            }
        }
        
        // Test with edge cases if we have more data
        if (offset < Size) {
            uint8_t edge_case = Data[offset++] % 4;
            
            // Create a non-inplace ReLU for edge case testing
            torch::nn::ReLU relu_safe(torch::nn::ReLUOptions().inplace(false));
            
            if (edge_case == 0 && input.numel() > 0) {
                // Test with all negative values
                torch::Tensor neg_input = -torch::abs(input);
                torch::Tensor neg_output = relu_safe->forward(neg_input);
            } else if (edge_case == 1) {
                // Test with NaN values if floating point
                if (input.is_floating_point() && input.numel() > 0) {
                    torch::Tensor nan_input = input.clone();
                    nan_input.index_put_({0}, std::numeric_limits<float>::quiet_NaN());
                    torch::Tensor nan_output = relu_safe->forward(nan_input);
                }
            } else if (edge_case == 2) {
                // Test with infinity values if floating point
                if (input.is_floating_point() && input.numel() > 0) {
                    torch::Tensor inf_input = input.clone();
                    inf_input.index_put_({0}, std::numeric_limits<float>::infinity());
                    torch::Tensor inf_output = relu_safe->forward(inf_input);
                }
            } else {
                // Test with very large values
                if (input.is_floating_point() && input.numel() > 0) {
                    torch::Tensor large_input = input.clone();
                    large_input.index_put_({0}, 1e38f);
                    torch::Tensor large_output = relu_safe->forward(large_input);
                }
            }
        }
        
        // Additional coverage: test LeakyReLU and PReLU variants
        if (offset < Size) {
            uint8_t extra_variant = Data[offset++] % 3;
            
            if (extra_variant == 0) {
                // LeakyReLU with fuzzed negative slope
                float negative_slope = 0.01f;
                if (offset + sizeof(float) <= Size) {
                    negative_slope = std::abs(*reinterpret_cast<const float*>(Data + offset)) / 100.0f;
                    offset += sizeof(float);
                    // Clamp to reasonable range
                    negative_slope = std::min(1.0f, std::max(0.0f, negative_slope));
                }
                torch::nn::LeakyReLU leaky(torch::nn::LeakyReLUOptions().negative_slope(negative_slope));
                torch::Tensor leaky_output = leaky->forward(input);
            } else if (extra_variant == 1) {
                // Test ELU
                torch::nn::ELU elu(torch::nn::ELUOptions().alpha(1.0));
                torch::Tensor elu_output = elu->forward(input);
            } else {
                // Test SELU
                torch::nn::SELU selu(torch::nn::SELUOptions().inplace(false));
                torch::Tensor selu_output = selu->forward(input);
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