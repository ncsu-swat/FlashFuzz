#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
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
        // Need at least enough bytes for parameters
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract RNN parameters from the input data
        // Extract input_size (1-32)
        int64_t input_size = (Data[offset++] % 32) + 1;
        
        // Extract hidden_size (1-32)
        int64_t hidden_size = (Data[offset++] % 32) + 1;
        
        // Extract num_layers (1-3)
        int64_t num_layers = (Data[offset++] % 3) + 1;
        
        // Extract flags
        bool bias = Data[offset++] % 2 == 0;
        bool batch_first = Data[offset++] % 2 == 0;
        bool bidirectional = Data[offset++] % 2 == 0;
        
        // Extract dropout (only valid if num_layers > 1)
        double dropout = 0.0;
        if (num_layers > 1 && offset < Size) {
            dropout = (Data[offset++] % 10) / 20.0; // 0.0 to 0.45
        } else {
            offset++; // consume the byte anyway
        }
        
        // Extract sequence length and batch size for input tensor
        int64_t seq_len = (Data[offset++] % 16) + 1;
        int64_t batch_size = (Data[offset++] % 8) + 1;
        
        // Choose RNN type: 0=RNN_TANH, 1=RNN_RELU, 2=LSTM, 3=GRU
        uint8_t rnn_type = Data[offset++] % 4;
        
        // Create input tensor with proper shape
        // Shape: [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
        torch::Tensor input_tensor;
        if (batch_first) {
            input_tensor = torch::randn({batch_size, seq_len, input_size});
        } else {
            input_tensor = torch::randn({seq_len, batch_size, input_size});
        }
        
        // Modulate input tensor with fuzzer data if available
        if (offset + 4 <= Size) {
            float scale;
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(scale) && std::abs(scale) < 100.0f) {
                input_tensor = input_tensor * scale;
            }
        }
        
        int64_t num_directions = bidirectional ? 2 : 1;
        
        try {
            if (rnn_type == 0) {
                // RNN with tanh
                auto options = torch::nn::RNNOptions(input_size, hidden_size)
                    .num_layers(num_layers)
                    .bias(bias)
                    .batch_first(batch_first)
                    .dropout(dropout)
                    .bidirectional(bidirectional)
                    .nonlinearity(torch::kTanh);
                
                torch::nn::RNN rnn(options);
                
                // Create initial hidden state
                torch::Tensor h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
                
                // Forward pass
                auto output = rnn->forward(input_tensor, h0);
                
                // Access output to ensure computation
                auto out_tensor = std::get<0>(output);
                auto hn = std::get<1>(output);
                (void)out_tensor.sum().item<float>();
                (void)hn.sum().item<float>();
                
            } else if (rnn_type == 1) {
                // RNN with relu
                auto options = torch::nn::RNNOptions(input_size, hidden_size)
                    .num_layers(num_layers)
                    .bias(bias)
                    .batch_first(batch_first)
                    .dropout(dropout)
                    .bidirectional(bidirectional)
                    .nonlinearity(torch::kReLU);
                
                torch::nn::RNN rnn(options);
                
                // Create initial hidden state
                torch::Tensor h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
                
                // Forward pass
                auto output = rnn->forward(input_tensor, h0);
                
                // Access output to ensure computation
                auto out_tensor = std::get<0>(output);
                auto hn = std::get<1>(output);
                (void)out_tensor.sum().item<float>();
                (void)hn.sum().item<float>();
                
            } else if (rnn_type == 2) {
                // LSTM
                auto options = torch::nn::LSTMOptions(input_size, hidden_size)
                    .num_layers(num_layers)
                    .bias(bias)
                    .batch_first(batch_first)
                    .dropout(dropout)
                    .bidirectional(bidirectional);
                
                torch::nn::LSTM lstm(options);
                
                // Create initial hidden state and cell state
                torch::Tensor h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
                torch::Tensor c0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
                
                // Forward pass
                auto output = lstm->forward(input_tensor, std::make_tuple(h0, c0));
                
                // Access output to ensure computation
                auto out_tensor = std::get<0>(output);
                auto hidden_tuple = std::get<1>(output);
                (void)out_tensor.sum().item<float>();
                (void)std::get<0>(hidden_tuple).sum().item<float>();
                (void)std::get<1>(hidden_tuple).sum().item<float>();
                
            } else {
                // GRU
                auto options = torch::nn::GRUOptions(input_size, hidden_size)
                    .num_layers(num_layers)
                    .bias(bias)
                    .batch_first(batch_first)
                    .dropout(dropout)
                    .bidirectional(bidirectional);
                
                torch::nn::GRU gru(options);
                
                // Create initial hidden state
                torch::Tensor h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
                
                // Forward pass
                auto output = gru->forward(input_tensor, h0);
                
                // Access output to ensure computation
                auto out_tensor = std::get<0>(output);
                auto hn = std::get<1>(output);
                (void)out_tensor.sum().item<float>();
                (void)hn.sum().item<float>();
            }
        } catch (const c10::Error& e) {
            // Silently catch PyTorch-specific errors (expected for some parameter combinations)
        } catch (const std::runtime_error& e) {
            // Silently catch runtime errors from invalid configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}