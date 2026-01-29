#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        if (Size < 16) return 0; // Need minimum data for parameters and tensor
        
        // Get parameters for RNN from the input data first
        uint8_t input_size_byte = Data[offset++];
        int64_t input_size = (input_size_byte % 16) + 1; // Between 1 and 16
        
        uint8_t hidden_size_byte = Data[offset++];
        int64_t hidden_size = (hidden_size_byte % 32) + 1; // Between 1 and 32
        
        uint8_t num_layers_byte = Data[offset++];
        int64_t num_layers = (num_layers_byte % 4) + 1; // Between 1 and 4 layers
        
        uint8_t bias_byte = Data[offset++];
        bool bias = bias_byte % 2 == 1;
        
        uint8_t batch_first_byte = Data[offset++];
        bool batch_first = batch_first_byte % 2 == 1;
        
        uint8_t dropout_byte = Data[offset++];
        // Dropout only applies when num_layers > 1
        double dropout = (num_layers > 1) ? (dropout_byte / 510.0) : 0.0; // Between 0 and 0.5
        
        uint8_t bidirectional_byte = Data[offset++];
        bool bidirectional = bidirectional_byte % 2 == 1;
        
        uint8_t rnn_type_byte = Data[offset++];
        int rnn_type = rnn_type_byte % 3; // 0: RNN, 1: LSTM, 2: GRU
        
        uint8_t seq_len_byte = Data[offset++];
        int64_t seq_len = (seq_len_byte % 16) + 1; // Between 1 and 16
        
        uint8_t batch_size_byte = Data[offset++];
        int64_t batch_size = (batch_size_byte % 8) + 1; // Between 1 and 8
        
        // Create input tensor with proper shape for RNN
        // Shape: [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
        torch::Tensor input;
        if (batch_first) {
            input = torch::randn({batch_size, seq_len, input_size});
        } else {
            input = torch::randn({seq_len, batch_size, input_size});
        }
        
        // Use remaining data to perturb the input tensor
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t num_elements = std::min(remaining / sizeof(float), static_cast<size_t>(input.numel()));
            if (num_elements > 0) {
                auto input_accessor = input.accessor<float, 3>();
                for (size_t i = 0; i < num_elements && offset + sizeof(float) <= Size; i++) {
                    float val;
                    memcpy(&val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    // Clamp to reasonable range to avoid NaN issues
                    if (std::isfinite(val)) {
                        val = std::max(-10.0f, std::min(10.0f, val));
                        int64_t idx = i;
                        int64_t d0 = input.size(0);
                        int64_t d1 = input.size(1);
                        int64_t d2 = input.size(2);
                        int64_t i0 = idx / (d1 * d2);
                        int64_t i1 = (idx % (d1 * d2)) / d2;
                        int64_t i2 = idx % d2;
                        if (i0 < d0 && i1 < d1 && i2 < d2) {
                            input_accessor[i0][i1][i2] = val;
                        }
                    }
                }
            }
        }
        
        // Create initial hidden state
        int64_t num_directions = bidirectional ? 2 : 1;
        torch::Tensor h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
        
        // Apply the RNN operation based on type
        try {
            if (rnn_type == 0) { // RNN
                torch::nn::RNNOptions rnn_options = torch::nn::RNNOptions(input_size, hidden_size)
                    .num_layers(num_layers)
                    .bias(bias)
                    .batch_first(batch_first)
                    .dropout(dropout)
                    .bidirectional(bidirectional);
                    
                torch::nn::RNN rnn_module(rnn_options);
                rnn_module->eval(); // Set to eval mode to disable dropout randomness
                auto output = rnn_module->forward(input, h0);
                // Access output to ensure computation happens
                auto out_tensor = std::get<0>(output);
                auto out_hidden = std::get<1>(output);
                (void)out_tensor.sum().item<float>();
                
            } else if (rnn_type == 1) { // LSTM
                torch::nn::LSTMOptions lstm_options = torch::nn::LSTMOptions(input_size, hidden_size)
                    .num_layers(num_layers)
                    .bias(bias)
                    .batch_first(batch_first)
                    .dropout(dropout)
                    .bidirectional(bidirectional);
                    
                torch::Tensor c0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
                
                torch::nn::LSTM lstm_module(lstm_options);
                lstm_module->eval();
                auto output = lstm_module->forward(input, std::make_tuple(h0, c0));
                auto out_tensor = std::get<0>(output);
                (void)out_tensor.sum().item<float>();
                
            } else { // GRU
                torch::nn::GRUOptions gru_options = torch::nn::GRUOptions(input_size, hidden_size)
                    .num_layers(num_layers)
                    .bias(bias)
                    .batch_first(batch_first)
                    .dropout(dropout)
                    .bidirectional(bidirectional);
                    
                torch::nn::GRU gru_module(gru_options);
                gru_module->eval();
                auto output = gru_module->forward(input, h0);
                auto out_tensor = std::get<0>(output);
                (void)out_tensor.sum().item<float>();
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors (shape mismatches, etc.) are expected
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}