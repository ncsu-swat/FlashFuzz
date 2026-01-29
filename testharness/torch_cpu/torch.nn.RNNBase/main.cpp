#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 12) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse RNN parameters from the data
        uint8_t input_size = (Data[offset++] % 16) + 1;      // 1-16
        uint8_t hidden_size = (Data[offset++] % 32) + 1;     // 1-32
        uint8_t num_layers = (Data[offset++] % 3) + 1;       // 1-3
        uint8_t seq_len = (Data[offset++] % 16) + 1;         // 1-16
        uint8_t batch_size = (Data[offset++] % 8) + 1;       // 1-8
        bool bidirectional = Data[offset++] % 2 == 1;
        bool batch_first = Data[offset++] % 2 == 1;
        uint8_t rnn_type_val = Data[offset++] % 4;
        
        // Dropout only applies when num_layers > 1
        float dropout = 0.0f;
        if (num_layers > 1 && offset < Size) {
            dropout = static_cast<float>(Data[offset++] % 50) / 100.0f; // 0.0 to 0.49
        }
        
        // Determine RNN type
        enum RNNType { RNN_TANH, RNN_RELU, LSTM, GRU };
        RNNType rnn_type = static_cast<RNNType>(rnn_type_val);
        
        // Create input tensor with proper 3D shape for RNN
        torch::Tensor input;
        if (batch_first) {
            input = torch::randn({batch_size, seq_len, input_size});
        } else {
            input = torch::randn({seq_len, batch_size, input_size});
        }
        
        int64_t num_directions = bidirectional ? 2 : 1;
        
        // Create RNN module and run forward pass
        try {
            if (rnn_type == RNN_TANH || rnn_type == RNN_RELU) {
                torch::nn::RNNOptions options = torch::nn::RNNOptions(input_size, hidden_size)
                    .num_layers(num_layers)
                    .dropout(dropout)
                    .bidirectional(bidirectional)
                    .batch_first(batch_first);
                
                // Set nonlinearity based on rnn_type - must be done separately due to different enum types
                if (rnn_type == RNN_TANH) {
                    options.nonlinearity(torch::kTanh);
                } else {
                    options.nonlinearity(torch::kReLU);
                }
                
                torch::nn::RNN rnn(options);
                rnn->eval(); // Set to eval mode to make dropout deterministic
                
                // Forward without hidden state
                auto output1 = rnn->forward(input);
                
                // Forward with initial hidden state
                torch::Tensor h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
                auto output2 = rnn->forward(input, h0);
                
            } else if (rnn_type == LSTM) {
                torch::nn::LSTMOptions options = torch::nn::LSTMOptions(input_size, hidden_size)
                    .num_layers(num_layers)
                    .dropout(dropout)
                    .bidirectional(bidirectional)
                    .batch_first(batch_first);
                
                torch::nn::LSTM lstm(options);
                lstm->eval();
                
                // Forward without hidden state
                auto output1 = lstm->forward(input);
                
                // Forward with initial hidden state (h0, c0)
                torch::Tensor h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
                torch::Tensor c0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
                auto output2 = lstm->forward(input, std::make_tuple(h0, c0));
                
            } else if (rnn_type == GRU) {
                torch::nn::GRUOptions options = torch::nn::GRUOptions(input_size, hidden_size)
                    .num_layers(num_layers)
                    .dropout(dropout)
                    .bidirectional(bidirectional)
                    .batch_first(batch_first);
                
                torch::nn::GRU gru(options);
                gru->eval();
                
                // Forward without hidden state
                auto output1 = gru->forward(input);
                
                // Forward with initial hidden state
                torch::Tensor h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
                auto output2 = gru->forward(input, h0);
            }
        } catch (const c10::Error& e) {
            // Expected errors from invalid configurations
        }
        
        // Test with different tensor values from fuzzer data
        if (offset + 4 <= Size) {
            try {
                torch::Tensor fuzz_input;
                if (batch_first) {
                    fuzz_input = fuzzer_utils::createTensor(Data, Size, offset);
                    // Reshape to 3D if possible
                    if (fuzz_input.numel() >= input_size) {
                        int64_t total = fuzz_input.numel();
                        int64_t new_seq = std::max(1L, total / (batch_size * input_size));
                        fuzz_input = fuzz_input.flatten().narrow(0, 0, new_seq * batch_size * input_size)
                                              .reshape({batch_size, new_seq, input_size});
                    } else {
                        fuzz_input = torch::randn({batch_size, seq_len, input_size});
                    }
                } else {
                    fuzz_input = fuzzer_utils::createTensor(Data, Size, offset);
                    if (fuzz_input.numel() >= input_size) {
                        int64_t total = fuzz_input.numel();
                        int64_t new_seq = std::max(1L, total / (batch_size * input_size));
                        fuzz_input = fuzz_input.flatten().narrow(0, 0, new_seq * batch_size * input_size)
                                              .reshape({new_seq, batch_size, input_size});
                    } else {
                        fuzz_input = torch::randn({seq_len, batch_size, input_size});
                    }
                }
                
                torch::nn::RNNOptions options = torch::nn::RNNOptions(input_size, hidden_size)
                    .num_layers(1)
                    .batch_first(batch_first);
                torch::nn::RNN rnn(options);
                rnn->eval();
                auto output = rnn->forward(fuzz_input);
                
            } catch (const c10::Error& e) {
                // Expected errors from shape mismatches
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