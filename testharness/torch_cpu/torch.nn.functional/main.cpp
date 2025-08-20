#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create tensors and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create parameters for nn.functional operations
        // We'll test various nn.functional operations based on available bytes
        if (offset + 1 >= Size) {
            return 0;
        }
        
        // Use a byte to select which nn.functional operation to test
        uint8_t op_selector = Data[offset++];
        
        // Test different nn.functional operations based on the selector
        switch (op_selector % 10) {
            case 0: {
                // Test relu
                auto output = torch::nn::functional::relu(input);
                break;
            }
            case 1: {
                // Test sigmoid
                auto output = torch::sigmoid(input);
                break;
            }
            case 2: {
                // Test tanh
                auto output = torch::tanh(input);
                break;
            }
            case 3: {
                // Test softmax
                if (offset + 1 >= Size) {
                    break;
                }
                int64_t dim = static_cast<int64_t>(Data[offset++]);
                auto output = torch::nn::functional::softmax(input, torch::nn::functional::SoftmaxFuncOptions(dim));
                break;
            }
            case 4: {
                // Test dropout
                if (offset + sizeof(float) >= Size) {
                    break;
                }
                float p = 0.0;
                std::memcpy(&p, Data + offset, sizeof(float));
                offset += sizeof(float);
                p = std::abs(p) / 10.0f; // Normalize to [0, 0.1]
                auto output = torch::dropout(input, p, true);
                break;
            }
            case 5: {
                // Test log_softmax
                if (offset + 1 >= Size) {
                    break;
                }
                int64_t dim = static_cast<int64_t>(Data[offset++]);
                auto output = torch::nn::functional::log_softmax(input, torch::nn::functional::LogSoftmaxFuncOptions(dim));
                break;
            }
            case 6: {
                // Test leaky_relu
                if (offset + sizeof(float) >= Size) {
                    break;
                }
                float negative_slope = 0.0;
                std::memcpy(&negative_slope, Data + offset, sizeof(float));
                offset += sizeof(float);
                negative_slope = std::abs(negative_slope) / 10.0f; // Normalize to [0, 0.1]
                auto output = torch::nn::functional::leaky_relu(input, torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope));
                break;
            }
            case 7: {
                // Test gelu
                auto output = torch::nn::functional::gelu(input);
                break;
            }
            case 8: {
                // Test batch_norm
                if (input.dim() < 2 || offset + 2 >= Size) {
                    break;
                }
                
                // Create running_mean and running_var tensors
                torch::Tensor running_mean = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor running_var = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Get momentum and eps parameters
                float momentum = 0.1;
                float eps = 1e-5;
                if (offset + 2*sizeof(float) <= Size) {
                    std::memcpy(&momentum, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    std::memcpy(&eps, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    
                    // Ensure valid values
                    momentum = std::abs(momentum) / 10.0f; // Normalize to [0, 0.1]
                    eps = std::abs(eps) / 1000.0f + 1e-6;  // Ensure positive and small
                }
                
                auto options = torch::nn::functional::BatchNormFuncOptions()
                    .training(true)
                    .momentum(momentum)
                    .eps(eps);
                
                auto output = torch::nn::functional::batch_norm(
                    input, running_mean, running_var, options);
                break;
            }
            case 9: {
                // Test layer_norm
                if (input.dim() < 1 || offset + 1 >= Size) {
                    break;
                }
                
                // Create normalized_shape
                std::vector<int64_t> normalized_shape;
                uint8_t num_dims = Data[offset++] % 4 + 1; // 1 to 4 dimensions
                
                for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                    int64_t dim_size = static_cast<int64_t>(Data[offset++]) % 16 + 1; // 1 to 16
                    normalized_shape.push_back(dim_size);
                }
                
                // Create weight and bias tensors
                torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor bias = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Get eps parameter
                float eps = 1e-5;
                if (offset + sizeof(float) <= Size) {
                    std::memcpy(&eps, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    eps = std::abs(eps) / 1000.0f + 1e-6;  // Ensure positive and small
                }
                
                auto output = torch::layer_norm(input, normalized_shape, weight, bias, eps);
                break;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}