#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) return 0; // Need minimum data for basic parameters
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get parameters for RNN from the input data
        uint8_t hidden_size_byte = (offset < Size) ? Data[offset++] : 1;
        int64_t hidden_size = (hidden_size_byte % 32) + 1; // Ensure positive hidden size
        
        uint8_t num_layers_byte = (offset < Size) ? Data[offset++] : 1;
        int64_t num_layers = (num_layers_byte % 4) + 1; // Between 1 and 4 layers
        
        uint8_t bias_byte = (offset < Size) ? Data[offset++] : 0;
        bool bias = bias_byte % 2 == 1; // 0 or 1
        
        uint8_t batch_first_byte = (offset < Size) ? Data[offset++] : 0;
        bool batch_first = batch_first_byte % 2 == 1; // 0 or 1
        
        uint8_t dropout_byte = (offset < Size) ? Data[offset++] : 0;
        double dropout = dropout_byte / 255.0; // Between 0 and 1
        
        uint8_t bidirectional_byte = (offset < Size) ? Data[offset++] : 0;
        bool bidirectional = bidirectional_byte % 2 == 1; // 0 or 1
        
        uint8_t rnn_type_byte = (offset < Size) ? Data[offset++] : 0;
        int rnn_type = rnn_type_byte % 3; // 0: RNN, 1: LSTM, 2: GRU
        
        // Get input size from the input tensor
        int64_t input_size = 1;
        if (input.dim() > 0 && input.size(input.dim() - 1) > 0) {
            input_size = input.size(input.dim() - 1);
        }
        
        // Create RNN module based on the type
        torch::nn::RNNOptions rnn_options = torch::nn::RNNOptions(input_size, hidden_size)
            .num_layers(num_layers)
            .bias(bias)
            .batch_first(batch_first)
            .dropout(dropout)
            .bidirectional(bidirectional);
            
        torch::nn::LSTMOptions lstm_options = torch::nn::LSTMOptions(input_size, hidden_size)
            .num_layers(num_layers)
            .bias(bias)
            .batch_first(batch_first)
            .dropout(dropout)
            .bidirectional(bidirectional);
            
        torch::nn::GRUOptions gru_options = torch::nn::GRUOptions(input_size, hidden_size)
            .num_layers(num_layers)
            .bias(bias)
            .batch_first(batch_first)
            .dropout(dropout)
            .bidirectional(bidirectional);
        
        // Reshape input tensor if needed to match expected RNN input shape
        // RNN expects [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first=true
        if (input.dim() == 0) {
            input = input.reshape({1, 1, 1});
        } else if (input.dim() == 1) {
            input = input.reshape({1, 1, input.size(0)});
        } else if (input.dim() == 2) {
            input = input.reshape({input.size(0), 1, input.size(1)});
        } else if (input.dim() > 3) {
            // Keep only the first 3 dimensions
            std::vector<int64_t> new_shape = {input.size(0), input.size(1), input.size(2)};
            input = input.reshape(new_shape);
        }
        
        // Create initial hidden state
        torch::Tensor h0;
        torch::Tensor c0;
        int64_t num_directions = bidirectional ? 2 : 1;
        
        if (rnn_type == 1) { // LSTM needs both h0 and c0
            h0 = torch::zeros({num_layers * num_directions, input.size(batch_first ? 0 : 1), hidden_size});
            c0 = torch::zeros({num_layers * num_directions, input.size(batch_first ? 0 : 1), hidden_size});
        } else { // RNN and GRU only need h0
            h0 = torch::zeros({num_layers * num_directions, input.size(batch_first ? 0 : 1), hidden_size});
        }
        
        // Apply the RNN operation
        try {
            if (rnn_type == 0) { // RNN
                torch::nn::RNN rnn_module(rnn_options);
                auto output = rnn_module->forward(input, h0);
            } else if (rnn_type == 1) { // LSTM
                torch::nn::LSTM lstm_module(lstm_options);
                auto output = lstm_module->forward(input, std::make_tuple(h0, c0));
            } else { // GRU
                torch::nn::GRU gru_module(gru_options);
                auto output = gru_module->forward(input, h0);
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and can be ignored
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