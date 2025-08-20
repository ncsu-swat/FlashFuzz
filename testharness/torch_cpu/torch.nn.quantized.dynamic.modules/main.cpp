#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get some configuration parameters from the input data
        uint8_t config_byte = (offset < Size) ? Data[offset++] : 0;
        
        // Create different types of dynamic quantized modules
        
        // Linear module
        try {
            int64_t in_features = std::max<int64_t>(1, input.size(0) % 100);
            int64_t out_features = std::max<int64_t>(1, (offset < Size ? Data[offset++] : 1) % 100);
            
            auto linear_options = torch::nn::LinearOptions(in_features, out_features);
            torch::nn::Linear linear_module(linear_options);
            
            // Reshape input if needed to match module requirements
            if (input.dim() < 2) {
                input = input.reshape({in_features, -1});
            } else if (input.size(0) != in_features) {
                input = input.reshape({in_features, -1});
            }
            
            // Apply the module
            auto output = linear_module(input);
        } catch (const std::exception &e) {
            // Continue with other modules
        }
        
        // LSTM module
        try {
            int64_t input_size = std::max<int64_t>(1, input.size(-1) % 50);
            int64_t hidden_size = std::max<int64_t>(1, (offset < Size ? Data[offset++] : 1) % 50);
            int64_t num_layers = std::max<int64_t>(1, (offset < Size ? Data[offset++] : 1) % 3);
            bool bias = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
            bool batch_first = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
            bool bidirectional = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
            
            auto lstm_options = torch::nn::LSTMOptions(input_size, hidden_size)
                .num_layers(num_layers)
                .bias(bias)
                .batch_first(batch_first)
                .bidirectional(bidirectional);
            torch::nn::LSTM lstm_module(lstm_options);
            
            // Reshape input if needed to match LSTM requirements
            // LSTM expects [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
            if (input.dim() < 3) {
                int64_t seq_len = std::max<int64_t>(1, input.numel() / (input_size * 2));
                int64_t batch = 2;
                
                if (batch_first) {
                    input = input.reshape({batch, seq_len, input_size});
                } else {
                    input = input.reshape({seq_len, batch, input_size});
                }
            }
            
            // Apply the module
            auto output = lstm_module(input);
        } catch (const std::exception &e) {
            // Continue with other modules
        }
        
        // GRU module
        try {
            int64_t input_size = std::max<int64_t>(1, input.size(-1) % 50);
            int64_t hidden_size = std::max<int64_t>(1, (offset < Size ? Data[offset++] : 1) % 50);
            int64_t num_layers = std::max<int64_t>(1, (offset < Size ? Data[offset++] : 1) % 3);
            bool bias = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
            bool batch_first = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
            bool bidirectional = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
            
            auto gru_options = torch::nn::GRUOptions(input_size, hidden_size)
                .num_layers(num_layers)
                .bias(bias)
                .batch_first(batch_first)
                .bidirectional(bidirectional);
            torch::nn::GRU gru_module(gru_options);
            
            // Reshape input if needed to match GRU requirements
            // GRU expects [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
            if (input.dim() < 3) {
                int64_t seq_len = std::max<int64_t>(1, input.numel() / (input_size * 2));
                int64_t batch = 2;
                
                if (batch_first) {
                    input = input.reshape({batch, seq_len, input_size});
                } else {
                    input = input.reshape({seq_len, batch, input_size});
                }
            }
            
            // Apply the module
            auto output = gru_module(input);
        } catch (const std::exception &e) {
            // Continue with other modules
        }
        
        // RNNCell module
        try {
            int64_t input_size = std::max<int64_t>(1, input.size(-1) % 50);
            int64_t hidden_size = std::max<int64_t>(1, (offset < Size ? Data[offset++] : 1) % 50);
            bool bias = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
            
            auto rnn_cell_options = torch::nn::RNNCellOptions(input_size, hidden_size)
                .bias(bias);
            torch::nn::RNNCell rnn_cell(rnn_cell_options);
            
            // Reshape input if needed to match RNNCell requirements
            // RNNCell expects [batch, input_size]
            if (input.dim() != 2) {
                int64_t batch = std::max<int64_t>(1, input.numel() / input_size);
                input = input.reshape({batch, input_size});
            }
            
            // Create hidden state
            auto h0 = torch::zeros({input.size(0), hidden_size});
            
            // Apply the module
            auto output = rnn_cell(input, h0);
        } catch (const std::exception &e) {
            // Continue with other modules
        }
        
        // LSTMCell module
        try {
            int64_t input_size = std::max<int64_t>(1, input.size(-1) % 50);
            int64_t hidden_size = std::max<int64_t>(1, (offset < Size ? Data[offset++] : 1) % 50);
            bool bias = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
            
            auto lstm_cell_options = torch::nn::LSTMCellOptions(input_size, hidden_size)
                .bias(bias);
            torch::nn::LSTMCell lstm_cell(lstm_cell_options);
            
            // Reshape input if needed to match LSTMCell requirements
            // LSTMCell expects [batch, input_size]
            if (input.dim() != 2) {
                int64_t batch = std::max<int64_t>(1, input.numel() / input_size);
                input = input.reshape({batch, input_size});
            }
            
            // Create hidden state and cell state
            auto h0 = torch::zeros({input.size(0), hidden_size});
            auto c0 = torch::zeros({input.size(0), hidden_size});
            
            // Apply the module
            auto output = lstm_cell(input, std::make_tuple(h0, c0));
        } catch (const std::exception &e) {
            // Continue with other modules
        }
        
        // GRUCell module
        try {
            int64_t input_size = std::max<int64_t>(1, input.size(-1) % 50);
            int64_t hidden_size = std::max<int64_t>(1, (offset < Size ? Data[offset++] : 1) % 50);
            bool bias = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
            
            auto gru_cell_options = torch::nn::GRUCellOptions(input_size, hidden_size)
                .bias(bias);
            torch::nn::GRUCell gru_cell(gru_cell_options);
            
            // Reshape input if needed to match GRUCell requirements
            // GRUCell expects [batch, input_size]
            if (input.dim() != 2) {
                int64_t batch = std::max<int64_t>(1, input.numel() / input_size);
                input = input.reshape({batch, input_size});
            }
            
            // Create hidden state
            auto h0 = torch::zeros({input.size(0), hidden_size});
            
            // Apply the module
            auto output = gru_cell(input, h0);
        } catch (const std::exception &e) {
            // Continue with other modules
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}