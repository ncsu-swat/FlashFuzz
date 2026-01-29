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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract module selector from the remaining data
        uint8_t module_selector = 0;
        if (offset < Size) {
            module_selector = Data[offset++];
        }
        
        // Create various NN modules based on the selector
        switch (module_selector % 10) {
            case 0: {
                // Linear layer
                try {
                    if (input.dim() >= 1 && input.size(-1) > 0) {
                        int64_t in_features = input.size(-1);
                        int64_t out_features = 10;
                        if (offset + 1 < Size) {
                            out_features = static_cast<int64_t>(Data[offset++]) % 32 + 1;
                        }
                        
                        torch::nn::Linear linear(in_features, out_features);
                        auto output = linear->forward(input);
                    }
                } catch (...) {
                    // Shape mismatches are expected
                }
                break;
            }
            
            case 1: {
                // Conv1d expects (N, C_in, L)
                try {
                    if (input.dim() == 3 && input.size(1) > 0 && input.size(2) > 0) {
                        int64_t in_channels = input.size(1);
                        int64_t out_channels = 16;
                        int64_t kernel_size = 3;
                        
                        if (offset + 1 < Size) {
                            out_channels = static_cast<int64_t>(Data[offset++]) % 32 + 1;
                        }
                        
                        if (offset + 1 < Size) {
                            kernel_size = static_cast<int64_t>(Data[offset++]) % 7 + 1;
                        }
                        
                        // Ensure kernel_size doesn't exceed input length
                        kernel_size = std::min(kernel_size, input.size(2));
                        if (kernel_size > 0) {
                            torch::nn::Conv1d conv(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size));
                            auto output = conv->forward(input);
                        }
                    }
                } catch (...) {
                    // Shape mismatches are expected
                }
                break;
            }
            
            case 2: {
                // Conv2d expects (N, C_in, H, W)
                try {
                    if (input.dim() == 4 && input.size(1) > 0 && input.size(2) > 0 && input.size(3) > 0) {
                        int64_t in_channels = input.size(1);
                        int64_t out_channels = 16;
                        int64_t kernel_size = 3;
                        
                        if (offset + 1 < Size) {
                            out_channels = static_cast<int64_t>(Data[offset++]) % 32 + 1;
                        }
                        
                        if (offset + 1 < Size) {
                            kernel_size = static_cast<int64_t>(Data[offset++]) % 7 + 1;
                        }
                        
                        // Ensure kernel_size doesn't exceed spatial dimensions
                        kernel_size = std::min(kernel_size, std::min(input.size(2), input.size(3)));
                        if (kernel_size > 0) {
                            torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size));
                            auto output = conv->forward(input);
                        }
                    }
                } catch (...) {
                    // Shape mismatches are expected
                }
                break;
            }
            
            case 3: {
                // Dropout - works on any shape
                try {
                    double p = 0.5;
                    if (offset + 1 < Size) {
                        p = static_cast<double>(Data[offset++]) / 255.0;
                    }
                    // Clamp p to valid range [0, 1)
                    p = std::min(p, 0.99);
                    
                    torch::nn::Dropout dropout(p);
                    auto output = dropout->forward(input);
                } catch (...) {
                    // Unexpected errors
                }
                break;
            }
            
            case 4: {
                // BatchNorm1d expects (N, C) or (N, C, L)
                try {
                    if ((input.dim() == 2 || input.dim() == 3) && input.size(1) > 0) {
                        int64_t num_features = input.size(1);
                        torch::nn::BatchNorm1d bn(num_features);
                        bn->eval();  // Set to eval mode to avoid running_mean/var issues with small batches
                        auto output = bn->forward(input);
                    }
                } catch (...) {
                    // Shape mismatches are expected
                }
                break;
            }
            
            case 5: {
                // BatchNorm2d expects (N, C, H, W)
                try {
                    if (input.dim() == 4 && input.size(1) > 0) {
                        int64_t num_features = input.size(1);
                        torch::nn::BatchNorm2d bn(num_features);
                        bn->eval();  // Set to eval mode
                        auto output = bn->forward(input);
                    }
                } catch (...) {
                    // Shape mismatches are expected
                }
                break;
            }
            
            case 6: {
                // RNN expects (L, N, H_in) or (N, L, H_in) with batch_first
                try {
                    if (input.dim() == 3 && input.size(-1) > 0 && input.size(0) > 0 && input.size(1) > 0) {
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
                } catch (...) {
                    // Shape mismatches are expected
                }
                break;
            }
            
            case 7: {
                // LSTM expects (L, N, H_in) or (N, L, H_in) with batch_first
                try {
                    if (input.dim() == 3 && input.size(-1) > 0 && input.size(0) > 0 && input.size(1) > 0) {
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
                } catch (...) {
                    // Shape mismatches are expected
                }
                break;
            }
            
            case 8: {
                // Embedding
                try {
                    int64_t num_embeddings = 100;
                    int64_t embedding_dim = 20;
                    
                    if (offset + 1 < Size) {
                        num_embeddings = static_cast<int64_t>(Data[offset++]) % 200 + 10;
                    }
                    
                    if (offset + 1 < Size) {
                        embedding_dim = static_cast<int64_t>(Data[offset++]) % 32 + 1;
                    }
                    
                    torch::nn::Embedding embedding(num_embeddings, embedding_dim);
                    
                    // Convert and clamp input to valid indices
                    auto long_input = input.to(torch::kLong).abs();
                    auto clamped_input = torch::remainder(long_input, num_embeddings);
                    auto output = embedding->forward(clamped_input);
                } catch (...) {
                    // Shape/type mismatches are expected
                }
                break;
            }
            
            case 9: {
                // Test activation functions
                uint8_t activation_selector = 0;
                if (offset < Size) {
                    activation_selector = Data[offset++];
                }
                
                try {
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
                            torch::nn::Tanh tanh_module;
                            auto output = tanh_module->forward(input);
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
                            if (input.dim() > 0) {
                                torch::nn::Softmax softmax(torch::nn::SoftmaxOptions(input.dim() - 1));
                                auto output = softmax->forward(input);
                            }
                            break;
                        }
                        case 5: {
                            if (input.dim() > 0) {
                                torch::nn::LogSoftmax log_softmax(torch::nn::LogSoftmaxOptions(input.dim() - 1));
                                auto output = log_softmax->forward(input);
                            }
                            break;
                        }
                    }
                } catch (...) {
                    // Activation function errors are expected for some inputs
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