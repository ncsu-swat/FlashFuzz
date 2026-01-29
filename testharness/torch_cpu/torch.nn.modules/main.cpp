#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

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
                try {
                    if (input_tensor.dim() < 1 || input_tensor.size(-1) < 1) {
                        break;
                    }
                    int64_t in_features = input_tensor.size(-1);
                    int64_t out_features = 10;
                    if (offset + 1 < Size) {
                        out_features = static_cast<int64_t>(Data[offset++]) % 100 + 1;
                    }
                    
                    torch::nn::Linear linear(in_features, out_features);
                    auto output = linear->forward(input_tensor);
                } catch (...) {
                    // Silently catch shape/dimension errors
                }
                break;
            }
            
            case 1: {
                // Conv1d module - expects (N, C, L)
                try {
                    int64_t total_elements = input_tensor.numel();
                    if (total_elements < 3) break;
                    
                    int64_t in_channels = 1;
                    int64_t seq_len = total_elements;
                    if (offset + 2 < Size) {
                        in_channels = static_cast<int64_t>(Data[offset++]) % 4 + 1;
                        seq_len = total_elements / in_channels;
                        if (seq_len < 1) {
                            in_channels = 1;
                            seq_len = total_elements;
                        }
                    }
                    
                    input_tensor = input_tensor.reshape({1, in_channels, seq_len});
                    
                    int64_t out_channels = 16;
                    int64_t kernel_size = 3;
                    
                    if (offset + 2 < Size) {
                        out_channels = static_cast<int64_t>(Data[offset++]) % 32 + 1;
                        kernel_size = static_cast<int64_t>(Data[offset++]) % std::min(seq_len, int64_t(7)) + 1;
                    }
                    kernel_size = std::min(kernel_size, seq_len);
                    if (kernel_size < 1) kernel_size = 1;
                    
                    torch::nn::Conv1d conv(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size));
                    auto output = conv->forward(input_tensor);
                } catch (...) {
                    // Silently catch shape/dimension errors
                }
                break;
            }
            
            case 2: {
                // Conv2d module - expects (N, C, H, W)
                try {
                    int64_t total_elements = input_tensor.numel();
                    if (total_elements < 4) break;
                    
                    int64_t in_channels = 1;
                    int64_t h = static_cast<int64_t>(std::sqrt(total_elements));
                    if (h < 1) h = 1;
                    int64_t w = total_elements / h;
                    if (w < 1) w = 1;
                    
                    // Reshape to fit available elements
                    int64_t actual_elements = h * w * in_channels;
                    input_tensor = input_tensor.flatten().narrow(0, 0, actual_elements).reshape({1, in_channels, h, w});
                    
                    int64_t out_channels = 16;
                    int64_t kernel_size = 3;
                    
                    if (offset + 2 < Size) {
                        out_channels = static_cast<int64_t>(Data[offset++]) % 32 + 1;
                        kernel_size = static_cast<int64_t>(Data[offset++]) % 5 + 1;
                    }
                    kernel_size = std::min(kernel_size, std::min(h, w));
                    if (kernel_size < 1) kernel_size = 1;
                    
                    torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size));
                    auto output = conv->forward(input_tensor);
                } catch (...) {
                    // Silently catch shape/dimension errors
                }
                break;
            }
            
            case 3: {
                // Dropout module
                try {
                    double p = 0.5;
                    if (offset < Size) {
                        p = static_cast<double>(Data[offset++]) / 255.0;
                    }
                    // Clamp p to valid range (0 <= p < 1)
                    p = std::max(0.0, std::min(p, 0.99));
                    
                    torch::nn::Dropout dropout(p);
                    auto output = dropout->forward(input_tensor);
                } catch (...) {
                    // Silently catch errors
                }
                break;
            }
            
            case 4: {
                // BatchNorm1d module - expects (N, C) or (N, C, L)
                try {
                    int64_t total_elements = input_tensor.numel();
                    if (total_elements < 2) break;
                    
                    int64_t num_features = std::min(total_elements, int64_t(64));
                    int64_t batch_size = total_elements / num_features;
                    if (batch_size < 1) batch_size = 1;
                    
                    int64_t actual_elements = batch_size * num_features;
                    input_tensor = input_tensor.flatten().narrow(0, 0, actual_elements).reshape({batch_size, num_features});
                    
                    torch::nn::BatchNorm1d bn(num_features);
                    bn->train(false);  // Use eval mode to avoid running stats issues with batch=1
                    auto output = bn->forward(input_tensor);
                } catch (...) {
                    // Silently catch errors
                }
                break;
            }
            
            case 5: {
                // RNN module - expects (L, N, H_in) for batch_first=false
                try {
                    int64_t total_elements = input_tensor.numel();
                    if (total_elements < 1) break;
                    
                    int64_t input_size = std::min(total_elements, int64_t(32));
                    int64_t seq_len = total_elements / input_size;
                    if (seq_len < 1) seq_len = 1;
                    
                    int64_t actual_elements = seq_len * input_size;
                    input_tensor = input_tensor.flatten().narrow(0, 0, actual_elements).reshape({seq_len, 1, input_size});
                    
                    int64_t hidden_size = 20;
                    int64_t num_layers = 1;
                    
                    if (offset + 2 < Size) {
                        hidden_size = static_cast<int64_t>(Data[offset++]) % 50 + 1;
                        num_layers = static_cast<int64_t>(Data[offset++]) % 3 + 1;
                    }
                    
                    torch::nn::RNN rnn(torch::nn::RNNOptions(input_size, hidden_size).num_layers(num_layers));
                    auto output = rnn->forward(input_tensor);
                } catch (...) {
                    // Silently catch errors
                }
                break;
            }
            
            case 6: {
                // LSTM module - expects (L, N, H_in) for batch_first=false
                try {
                    int64_t total_elements = input_tensor.numel();
                    if (total_elements < 1) break;
                    
                    int64_t input_size = std::min(total_elements, int64_t(32));
                    int64_t seq_len = total_elements / input_size;
                    if (seq_len < 1) seq_len = 1;
                    
                    int64_t actual_elements = seq_len * input_size;
                    input_tensor = input_tensor.flatten().narrow(0, 0, actual_elements).reshape({seq_len, 1, input_size});
                    
                    int64_t hidden_size = 20;
                    int64_t num_layers = 1;
                    
                    if (offset + 2 < Size) {
                        hidden_size = static_cast<int64_t>(Data[offset++]) % 50 + 1;
                        num_layers = static_cast<int64_t>(Data[offset++]) % 3 + 1;
                    }
                    
                    torch::nn::LSTM lstm(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers));
                    auto output = lstm->forward(input_tensor);
                } catch (...) {
                    // Silently catch errors
                }
                break;
            }
            
            case 7: {
                // Embedding module - expects LongTensor indices
                try {
                    int64_t num_embeddings = 100;
                    int64_t embedding_dim = 20;
                    
                    if (offset + 2 < Size) {
                        num_embeddings = static_cast<int64_t>(Data[offset++]) % 200 + 10;
                        embedding_dim = static_cast<int64_t>(Data[offset++]) % 50 + 1;
                    }
                    
                    // Convert to valid indices
                    input_tensor = input_tensor.abs().to(torch::kLong).remainder(num_embeddings);
                    
                    torch::nn::Embedding embedding(num_embeddings, embedding_dim);
                    auto output = embedding->forward(input_tensor);
                } catch (...) {
                    // Silently catch errors
                }
                break;
            }
            
            case 8: {
                // MaxPool2d module - expects (N, C, H, W)
                try {
                    int64_t total_elements = input_tensor.numel();
                    if (total_elements < 4) break;
                    
                    int64_t h = static_cast<int64_t>(std::sqrt(total_elements));
                    if (h < 2) h = 2;
                    int64_t w = total_elements / h;
                    if (w < 2) w = 2;
                    
                    int64_t actual_elements = h * w;
                    input_tensor = input_tensor.flatten().narrow(0, 0, actual_elements).reshape({1, 1, h, w});
                    
                    int64_t kernel_size = 2;
                    int64_t stride = 2;
                    
                    if (offset + 2 < Size) {
                        kernel_size = static_cast<int64_t>(Data[offset++]) % 4 + 1;
                        stride = static_cast<int64_t>(Data[offset++]) % 3 + 1;
                    }
                    kernel_size = std::min(kernel_size, std::min(h, w));
                    if (kernel_size < 1) kernel_size = 1;
                    if (stride < 1) stride = 1;
                    
                    torch::nn::MaxPool2d maxpool(torch::nn::MaxPool2dOptions(kernel_size).stride(stride));
                    auto output = maxpool->forward(input_tensor);
                } catch (...) {
                    // Silently catch errors
                }
                break;
            }
            
            case 9: {
                // Sequential module with multiple layers
                try {
                    int64_t total_elements = input_tensor.numel();
                    if (total_elements < 1) break;
                    
                    int64_t in_features = std::min(total_elements, int64_t(64));
                    int64_t batch_size = total_elements / in_features;
                    if (batch_size < 1) batch_size = 1;
                    
                    int64_t actual_elements = batch_size * in_features;
                    input_tensor = input_tensor.flatten().narrow(0, 0, actual_elements).reshape({batch_size, in_features});
                    
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
                } catch (...) {
                    // Silently catch errors
                }
                break;
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