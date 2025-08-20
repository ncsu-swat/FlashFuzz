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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a byte to determine which module to test
        uint8_t module_selector = 0;
        if (offset < Size) {
            module_selector = Data[offset++];
        }
        
        // Create various nn modules and test them
        switch (module_selector % 10) {
            case 0: {
                // Linear module
                int64_t in_features = input_tensor.size(-1);
                int64_t out_features = 10;
                if (offset + 1 < Size) {
                    out_features = static_cast<int64_t>(Data[offset++]) % 100 + 1;
                }
                
                torch::nn::Linear linear(in_features, out_features);
                auto output = linear->forward(input_tensor);
                break;
            }
            
            case 1: {
                // Conv1d module
                if (input_tensor.dim() < 2) {
                    input_tensor = input_tensor.reshape({1, 1, -1});
                } else if (input_tensor.dim() == 2) {
                    input_tensor = input_tensor.unsqueeze(0);
                }
                
                int64_t in_channels = input_tensor.size(1);
                int64_t out_channels = 16;
                int64_t kernel_size = 3;
                
                if (offset + 2 < Size) {
                    out_channels = static_cast<int64_t>(Data[offset++]) % 32 + 1;
                    kernel_size = static_cast<int64_t>(Data[offset++]) % 7 + 1;
                }
                
                torch::nn::Conv1d conv(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size));
                auto output = conv->forward(input_tensor);
                break;
            }
            
            case 2: {
                // Conv2d module
                if (input_tensor.dim() < 3) {
                    input_tensor = input_tensor.reshape({1, 1, 1, -1});
                } else if (input_tensor.dim() == 3) {
                    input_tensor = input_tensor.unsqueeze(0);
                }
                
                int64_t in_channels = input_tensor.size(1);
                int64_t out_channels = 16;
                int64_t kernel_size = 3;
                
                if (offset + 2 < Size) {
                    out_channels = static_cast<int64_t>(Data[offset++]) % 32 + 1;
                    kernel_size = static_cast<int64_t>(Data[offset++]) % 5 + 1;
                }
                
                torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size));
                auto output = conv->forward(input_tensor);
                break;
            }
            
            case 3: {
                // Dropout module
                double p = 0.5;
                if (offset < Size) {
                    p = static_cast<double>(Data[offset++]) / 255.0;
                }
                
                torch::nn::Dropout dropout(p);
                auto output = dropout->forward(input_tensor);
                break;
            }
            
            case 4: {
                // BatchNorm1d module
                if (input_tensor.dim() < 2) {
                    input_tensor = input_tensor.reshape({1, -1});
                }
                
                int64_t num_features = input_tensor.size(1);
                torch::nn::BatchNorm1d bn(num_features);
                auto output = bn->forward(input_tensor);
                break;
            }
            
            case 5: {
                // RNN module
                if (input_tensor.dim() < 2) {
                    input_tensor = input_tensor.reshape({1, 1, -1});
                } else if (input_tensor.dim() == 2) {
                    input_tensor = input_tensor.unsqueeze(1);
                }
                
                int64_t input_size = input_tensor.size(-1);
                int64_t hidden_size = 20;
                int64_t num_layers = 1;
                
                if (offset + 2 < Size) {
                    hidden_size = static_cast<int64_t>(Data[offset++]) % 50 + 1;
                    num_layers = static_cast<int64_t>(Data[offset++]) % 3 + 1;
                }
                
                torch::nn::RNN rnn(torch::nn::RNNOptions(input_size, hidden_size).num_layers(num_layers));
                auto output = rnn->forward(input_tensor);
                break;
            }
            
            case 6: {
                // LSTM module
                if (input_tensor.dim() < 2) {
                    input_tensor = input_tensor.reshape({1, 1, -1});
                } else if (input_tensor.dim() == 2) {
                    input_tensor = input_tensor.unsqueeze(1);
                }
                
                int64_t input_size = input_tensor.size(-1);
                int64_t hidden_size = 20;
                int64_t num_layers = 1;
                
                if (offset + 2 < Size) {
                    hidden_size = static_cast<int64_t>(Data[offset++]) % 50 + 1;
                    num_layers = static_cast<int64_t>(Data[offset++]) % 3 + 1;
                }
                
                torch::nn::LSTM lstm(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers));
                auto output = lstm->forward(input_tensor);
                break;
            }
            
            case 7: {
                // Embedding module
                if (input_tensor.dtype() != torch::kLong) {
                    input_tensor = input_tensor.to(torch::kLong).abs() % 100;
                }
                
                int64_t num_embeddings = 100;
                int64_t embedding_dim = 20;
                
                if (offset + 2 < Size) {
                    num_embeddings = static_cast<int64_t>(Data[offset++]) % 200 + 10;
                    embedding_dim = static_cast<int64_t>(Data[offset++]) % 50 + 1;
                }
                
                torch::nn::Embedding embedding(num_embeddings, embedding_dim);
                auto output = embedding->forward(input_tensor);
                break;
            }
            
            case 8: {
                // MaxPool2d module
                if (input_tensor.dim() < 3) {
                    input_tensor = input_tensor.reshape({1, 1, 1, -1});
                } else if (input_tensor.dim() == 3) {
                    input_tensor = input_tensor.unsqueeze(0);
                }
                
                int64_t kernel_size = 2;
                int64_t stride = 2;
                
                if (offset + 2 < Size) {
                    kernel_size = static_cast<int64_t>(Data[offset++]) % 4 + 1;
                    stride = static_cast<int64_t>(Data[offset++]) % 3 + 1;
                }
                
                torch::nn::MaxPool2d maxpool(torch::nn::MaxPool2dOptions(kernel_size).stride(stride));
                auto output = maxpool->forward(input_tensor);
                break;
            }
            
            case 9: {
                // Sequential module with multiple layers
                if (input_tensor.dim() < 2) {
                    input_tensor = input_tensor.reshape({1, -1});
                }
                
                int64_t in_features = input_tensor.size(-1);
                int64_t hidden_size = 20;
                int64_t out_features = 10;
                
                if (offset + 2 < Size) {
                    hidden_size = static_cast<int64_t>(Data[offset++]) % 50 + 1;
                    out_features = static_cast<int64_t>(Data[offset++]) % 20 + 1;
                }
                
                torch::nn::Sequential sequential(
                    torch::nn::Linear(in_features, hidden_size),
                    torch::nn::ReLU(),
                    torch::nn::Dropout(0.2),
                    torch::nn::Linear(hidden_size, out_features)
                );
                
                auto output = sequential->forward(input_tensor);
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