#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
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
                auto output = relu->forward(input);
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
                auto output = leaky_relu->forward(input);
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
                auto tanh = torch::nn::Tanh();
                auto output = tanh->forward(input);
                break;
            }
            case 6: {
                // Softmax
                auto softmax = torch::nn::Softmax(dim % (input.dim() + 1));
                auto output = softmax->forward(input);
                break;
            }
            case 7: {
                // LogSoftmax
                auto log_softmax = torch::nn::LogSoftmax(dim % (input.dim() + 1));
                auto output = log_softmax->forward(input);
                break;
            }
            case 8: {
                // ELU
                auto elu = torch::nn::ELU(torch::nn::ELUOptions().alpha(alpha));
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
                auto celu = torch::nn::CELU(torch::nn::CELUOptions().alpha(alpha));
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
                auto hardshrink = torch::nn::Hardshrink(torch::nn::HardshrinkOptions().lambda(alpha));
                auto output = hardshrink->forward(input);
                break;
            }
            case 13: {
                // Hardtanh
                auto hardtanh = torch::nn::Hardtanh(torch::nn::HardtanhOptions().min_val(-alpha).max_val(beta));
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
        return -1; // discard the input
    }
    return 0; // keep the input
}