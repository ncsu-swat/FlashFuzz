#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isnan, std::isinf

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a byte to determine which activation function to use
        uint8_t activation_type = 0;
        if (offset < Size) {
            activation_type = Data[offset++];
        }
        
        // Get parameters for activation functions that need them
        double alpha = 0.01;
        double beta = 1.0;
        int64_t dim = 0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize alpha - avoid NaN, Inf, and extreme values
            if (std::isnan(alpha) || std::isinf(alpha)) {
                alpha = 0.01;
            }
            alpha = std::max(-100.0, std::min(100.0, alpha));
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize beta - avoid NaN, Inf, and extreme values
            if (std::isnan(beta) || std::isinf(beta)) {
                beta = 1.0;
            }
            beta = std::max(-100.0, std::min(100.0, beta));
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply different activation functions based on activation_type
        switch (activation_type % 15) {
            case 0: {
                // ReLU
                auto relu = torch::nn::ReLU();
                auto output = relu->forward(input);
                break;
            }
            case 1: {
                // ReLU with inplace option
                bool inplace = offset < Size && (Data[offset++] & 0x01);
                auto relu = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(inplace));
                // Clone for inplace to avoid modifying original fuzzer tensor
                auto input_copy = inplace ? input.clone() : input;
                auto output = relu->forward(input_copy);
                break;
            }
            case 2: {
                // LeakyReLU
                auto leaky_relu = torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(alpha));
                auto output = leaky_relu->forward(input);
                break;
            }
            case 3: {
                // LeakyReLU with inplace option
                bool inplace = offset < Size && (Data[offset++] & 0x01);
                auto leaky_relu = torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(alpha).inplace(inplace));
                auto input_copy = inplace ? input.clone() : input;
                auto output = leaky_relu->forward(input_copy);
                break;
            }
            case 4: {
                // Sigmoid
                auto sigmoid = torch::nn::Sigmoid();
                auto output = sigmoid->forward(input);
                break;
            }
            case 5: {
                // Tanh
                auto tanh_module = torch::nn::Tanh();
                auto output = tanh_module->forward(input);
                break;
            }
            case 6: {
                // Softmax - requires at least 1 dimension
                if (input.dim() > 0) {
                    int64_t valid_dim = dim % input.dim();
                    if (valid_dim < 0) valid_dim += input.dim();
                    auto softmax = torch::nn::Softmax(torch::nn::SoftmaxOptions(valid_dim));
                    auto output = softmax->forward(input);
                }
                break;
            }
            case 7: {
                // LogSoftmax - requires at least 1 dimension
                if (input.dim() > 0) {
                    int64_t valid_dim = dim % input.dim();
                    if (valid_dim < 0) valid_dim += input.dim();
                    auto log_softmax = torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(valid_dim));
                    auto output = log_softmax->forward(input);
                }
                break;
            }
            case 8: {
                // ELU
                double elu_alpha = std::abs(alpha) + 0.001; // ELU alpha should be positive
                auto elu = torch::nn::ELU(torch::nn::ELUOptions().alpha(elu_alpha));
                auto output = elu->forward(input);
                break;
            }
            case 9: {
                // SELU
                auto selu = torch::nn::SELU();
                auto output = selu->forward(input);
                break;
            }
            case 10: {
                // CELU
                double celu_alpha = std::abs(alpha) + 0.001; // CELU alpha should be positive and non-zero
                auto celu = torch::nn::CELU(torch::nn::CELUOptions().alpha(celu_alpha));
                auto output = celu->forward(input);
                break;
            }
            case 11: {
                // GELU
                auto gelu = torch::nn::GELU();
                auto output = gelu->forward(input);
                break;
            }
            case 12: {
                // Hardshrink
                double lambda_val = std::abs(alpha); // lambda should be non-negative
                auto hardshrink = torch::nn::Hardshrink(torch::nn::HardshrinkOptions().lambda(lambda_val));
                auto output = hardshrink->forward(input);
                break;
            }
            case 13: {
                // Hardtanh - ensure min_val < max_val
                double min_val = std::min(alpha, beta);
                double max_val = std::max(alpha, beta);
                if (min_val == max_val) {
                    max_val = min_val + 1.0;
                }
                auto hardtanh = torch::nn::Hardtanh(torch::nn::HardtanhOptions().min_val(min_val).max_val(max_val));
                auto output = hardtanh->forward(input);
                break;
            }
            case 14: {
                // PReLU
                auto prelu = torch::nn::PReLU();
                auto output = prelu->forward(input);
                break;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}