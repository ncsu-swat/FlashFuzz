#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract some parameters for NN modules from the remaining data
        uint8_t module_selector = 0;
        if (offset < Size) {
            module_selector = Data[offset++];
        }
        
        // Create various NN modules based on the selector
        switch (module_selector % 10) {
            case 0: {
                // Linear layer
                int64_t in_features = input.size(-1);
                int64_t out_features = 10;
                if (offset + 1 < Size) {
                    out_features = static_cast<int64_t>(Data[offset++]) % 32 + 1;
                }
                
                torch::nn::Linear linear(in_features, out_features);
                
                // Apply the module if input has correct shape
                if (input.dim() >= 1) {
                    auto output = linear->forward(input);
                }
                break;
            }
            
            case 1: {
                // Conv1d
                if (input.dim() >= 2) {
                    int64_t in_channels = input.size(0);
                    int64_t out_channels = 16;
                    int64_t kernel_size = 3;
                    
                    if (offset + 1 < Size) {
                        out_channels = static_cast<int64_t>(Data[offset++]) % 32 + 1;
                    }
                    
                    if (offset + 1 < Size) {
                        kernel_size = static_cast<int64_t>(Data[offset++]) % 7 + 1;
                    }
                    
                    torch::nn::Conv1d conv(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size));
                    auto output = conv->forward(input);
                }
                break;
            }
            
            case 2: {
                // Conv2d
                if (input.dim() >= 3) {
                    int64_t in_channels = input.size(0);
                    int64_t out_channels = 16;
                    int64_t kernel_size = 3;
                    
                    if (offset + 1 < Size) {
                        out_channels = static_cast<int64_t>(Data[offset++]) % 32 + 1;
                    }
                    
                    if (offset + 1 < Size) {
                        kernel_size = static_cast<int64_t>(Data[offset++]) % 7 + 1;
                    }
                    
                    torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size));
                    auto output = conv->forward(input);
                }
                break;
            }
            
            case 3: {
                // Dropout
                double p = 0.5;
                if (offset + 1 < Size) {
                    p = static_cast<double>(Data[offset++]) / 255.0;
                }
                
                torch::nn::Dropout dropout(p);
                auto output = dropout->forward(input);
                break;
            }
            
            case 4: {
                // BatchNorm1d
                if (input.dim() >= 2) {
                    int64_t num_features = input.size(1);
                    torch::nn::BatchNorm1d bn(num_features);
                    auto output = bn->forward(input);
                }
                break;
            }
            
            case 5: {
                // BatchNorm2d
                if (input.dim() >= 3) {
                    int64_t num_features = input.size(1);
                    torch::nn::BatchNorm2d bn(num_features);
                    auto output = bn->forward(input);
                }
                break;
            }
            
            case 6: {
                // RNN
                if (input.dim() >= 2) {
                    int64_t input_size = input.size(-1);
                    int64_t hidden_size = 20;
                    int64_t num_layers = 1;
                    
                    if (offset + 1 < Size) {
                        hidden_size = static_cast<int64_t>(Data[offset++]) % 32 + 1;
                    }
                    
                    if (offset + 1 < Size) {
                        num_layers = static_cast<int64_t>(Data[offset++]) % 3 + 1;
                    }
                    
                    torch::nn::RNN rnn(torch::nn::RNNOptions(input_size, hidden_size).num_layers(num_layers));
                    auto output = rnn->forward(input);
                }
                break;
            }
            
            case 7: {
                // LSTM
                if (input.dim() >= 2) {
                    int64_t input_size = input.size(-1);
                    int64_t hidden_size = 20;
                    int64_t num_layers = 1;
                    
                    if (offset + 1 < Size) {
                        hidden_size = static_cast<int64_t>(Data[offset++]) % 32 + 1;
                    }
                    
                    if (offset + 1 < Size) {
                        num_layers = static_cast<int64_t>(Data[offset++]) % 3 + 1;
                    }
                    
                    torch::nn::LSTM lstm(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers));
                    auto output = lstm->forward(input);
                }
                break;
            }
            
            case 8: {
                // Embedding
                if (input.dtype() == torch::kLong || input.dtype() == torch::kInt) {
                    int64_t num_embeddings = 100;
                    int64_t embedding_dim = 20;
                    
                    if (offset + 1 < Size) {
                        num_embeddings = static_cast<int64_t>(Data[offset++]) % 200 + 10;
                    }
                    
                    if (offset + 1 < Size) {
                        embedding_dim = static_cast<int64_t>(Data[offset++]) % 32 + 1;
                    }
                    
                    torch::nn::Embedding embedding(num_embeddings, embedding_dim);
                    
                    // Ensure input values are within valid range
                    auto clamped_input = torch::clamp(input.to(torch::kLong), 0, num_embeddings - 1);
                    auto output = embedding->forward(clamped_input);
                }
                break;
            }
            
            case 9: {
                // Test activation functions
                uint8_t activation_selector = 0;
                if (offset < Size) {
                    activation_selector = Data[offset++];
                }
                
                switch (activation_selector % 6) {
                    case 0: {
                        torch::nn::ReLU relu;
                        auto output = relu->forward(input);
                        break;
                    }
                    case 1: {
                        torch::nn::Sigmoid sigmoid;
                        auto output = sigmoid->forward(input);
                        break;
                    }
                    case 2: {
                        torch::nn::Tanh tanh;
                        auto output = tanh->forward(input);
                        break;
                    }
                    case 3: {
                        double negative_slope = 0.01;
                        if (offset < Size) {
                            negative_slope = static_cast<double>(Data[offset++]) / 255.0;
                        }
                        torch::nn::LeakyReLU leaky_relu(torch::nn::LeakyReLUOptions().negative_slope(negative_slope));
                        auto output = leaky_relu->forward(input);
                        break;
                    }
                    case 4: {
                        torch::nn::Softmax softmax(input.dim() > 0 ? input.dim() - 1 : 0);
                        auto output = softmax->forward(input);
                        break;
                    }
                    case 5: {
                        torch::nn::LogSoftmax log_softmax(input.dim() > 0 ? input.dim() - 1 : 0);
                        auto output = log_softmax->forward(input);
                        break;
                    }
                }
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
