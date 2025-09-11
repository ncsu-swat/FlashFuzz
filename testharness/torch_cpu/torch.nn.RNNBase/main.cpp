#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse RNN parameters from the remaining data
        uint8_t hidden_size = 1;
        uint8_t num_layers = 1;
        bool bidirectional = false;
        bool batch_first = false;
        float dropout = 0.0f;
        
        if (offset + 4 <= Size) {
            hidden_size = Data[offset++] % 32 + 1;
            num_layers = Data[offset++] % 4 + 1;
            bidirectional = Data[offset++] % 2 == 1;
            batch_first = Data[offset++] % 2 == 1;
            
            if (offset + sizeof(float) <= Size) {
                float dropout_raw;
                std::memcpy(&dropout_raw, Data + offset, sizeof(float));
                offset += sizeof(float);
                dropout = std::abs(dropout_raw) / 10.0f; // Normalize to [0, 0.1]
            }
        }
        
        // Determine RNN type
        enum RNNType { RNN_TANH, RNN_RELU, LSTM, GRU };
        RNNType rnn_type = RNNType::RNN_TANH;
        
        if (offset < Size) {
            rnn_type = static_cast<RNNType>(Data[offset++] % 4);
        }
        
        // Get input size from the input tensor
        int64_t input_size = 1;
        if (input.dim() > 2) {
            input_size = input.size(-1);
        }
        
        // Create RNN module based on the type
        torch::nn::RNNOptions rnn_options = torch::nn::RNNOptions(input_size, hidden_size)
            .num_layers(num_layers)
            .dropout(dropout)
            .bidirectional(bidirectional)
            .batch_first(batch_first);
            
        torch::nn::GRUOptions gru_options = torch::nn::GRUOptions(input_size, hidden_size)
            .num_layers(num_layers)
            .dropout(dropout)
            .bidirectional(bidirectional)
            .batch_first(batch_first);
            
        torch::nn::LSTMOptions lstm_options = torch::nn::LSTMOptions(input_size, hidden_size)
            .num_layers(num_layers)
            .dropout(dropout)
            .bidirectional(bidirectional)
            .batch_first(batch_first);
        
        // Create RNN module and run forward pass
        try {
            if (rnn_type == RNN_TANH) {
                torch::nn::RNN rnn(rnn_options.nonlinearity(torch::kTanh));
                auto output = rnn->forward(input);
            } else if (rnn_type == RNN_RELU) {
                torch::nn::RNN rnn(rnn_options.nonlinearity(torch::kReLU));
                auto output = rnn->forward(input);
            } else if (rnn_type == LSTM) {
                torch::nn::LSTM lstm(lstm_options);
                auto output = lstm->forward(input);
            } else if (rnn_type == GRU) {
                torch::nn::GRU gru(gru_options);
                auto output = gru->forward(input);
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
            return 0;
        }
        
        // Try with initial hidden state
        try {
            if (rnn_type == RNN_TANH || rnn_type == RNN_RELU) {
                torch::nn::RNNOptions options = rnn_options.nonlinearity(rnn_type == RNN_TANH ? torch::kTanh : torch::kReLU);
                torch::nn::RNN rnn(options);
                
                // Create h0 with appropriate shape
                int64_t num_directions = bidirectional ? 2 : 1;
                torch::Tensor h0 = torch::zeros({num_layers * num_directions, 
                                                input.size(batch_first ? 0 : 1), 
                                                hidden_size});
                
                auto output = rnn->forward(input, h0);
            } else if (rnn_type == LSTM) {
                torch::nn::LSTM lstm(lstm_options);
                
                // Create h0 and c0 with appropriate shape
                int64_t num_directions = bidirectional ? 2 : 1;
                torch::Tensor h0 = torch::zeros({num_layers * num_directions, 
                                                input.size(batch_first ? 0 : 1), 
                                                hidden_size});
                torch::Tensor c0 = torch::zeros({num_layers * num_directions, 
                                                input.size(batch_first ? 0 : 1), 
                                                hidden_size});
                
                auto output = lstm->forward(input, std::make_tuple(h0, c0));
            } else if (rnn_type == GRU) {
                torch::nn::GRU gru(gru_options);
                
                // Create h0 with appropriate shape
                int64_t num_directions = bidirectional ? 2 : 1;
                torch::Tensor h0 = torch::zeros({num_layers * num_directions, 
                                                input.size(batch_first ? 0 : 1), 
                                                hidden_size});
                
                auto output = gru->forward(input, h0);
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
