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
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for LSTM from the data
        uint8_t input_size = (Data[offset++] % 16) + 1;    // 1-16
        uint8_t hidden_size = (Data[offset++] % 16) + 1;   // 1-16
        uint8_t num_layers = (Data[offset++] % 3) + 1;     // 1-3 layers
        uint8_t seq_len = (Data[offset++] % 8) + 1;        // 1-8
        uint8_t batch_size = (Data[offset++] % 4) + 1;     // 1-4
        bool bidirectional = Data[offset++] % 2;           // 0 or 1
        bool batch_first = Data[offset++] % 2;             // 0 or 1
        
        // Dropout only applies when num_layers > 1
        float dropout = 0.0f;
        if (num_layers > 1 && offset < Size) {
            dropout = static_cast<float>(Data[offset++] % 50) / 100.0f; // 0.0 to 0.49
        } else if (offset < Size) {
            offset++; // consume the byte anyway
        }
        
        // Create input tensor with correct shape for LSTM
        torch::Tensor input;
        if (batch_first) {
            input = torch::randn({batch_size, seq_len, input_size});
        } else {
            input = torch::randn({seq_len, batch_size, input_size});
        }
        
        // If there's more data, use it to modify the input values
        if (offset < Size) {
            size_t temp_offset = 0;
            torch::Tensor modifier = fuzzer_utils::createTensor(Data + offset, Size - offset, temp_offset);
            try {
                // Try to add some variation from fuzzer data
                if (modifier.numel() > 0) {
                    auto flat_modifier = modifier.flatten();
                    auto flat_input = input.flatten();
                    int64_t copy_size = std::min(flat_modifier.numel(), flat_input.numel());
                    flat_input.slice(0, 0, copy_size).copy_(flat_modifier.slice(0, 0, copy_size));
                }
            } catch (...) {
                // Ignore modifier errors
            }
        }
        
        // Create LSTM module
        auto lstm_options = torch::nn::LSTMOptions(input_size, hidden_size)
            .num_layers(num_layers)
            .bidirectional(bidirectional)
            .batch_first(batch_first);
        
        if (num_layers > 1) {
            lstm_options.dropout(dropout);
        }
        
        torch::nn::LSTM lstm(lstm_options);
        lstm->eval(); // Set to eval mode to disable dropout randomness
        
        int64_t num_directions = bidirectional ? 2 : 1;
        
        // Prepare initial hidden state (h0, c0)
        torch::Tensor h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
        torch::Tensor c0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
        
        // Run LSTM forward with initial states
        auto [output_seq, hidden_states] = lstm->forward(input, std::make_tuple(h0, c0));
        auto [h_n, c_n] = hidden_states;
        
        // Perform operations on the output to ensure it's used
        auto sum = output_seq.sum() + h_n.sum() + c_n.sum();
        (void)sum;
        
        // Also test forward without initial hidden state
        try {
            auto [output_seq2, hidden_states2] = lstm->forward(input);
            // output_seq2 is already a tensor from the structured binding, no need for std::get
            auto sum2 = output_seq2.sum();
            (void)sum2;
        } catch (...) {
            // Ignore errors from this variant
        }
        
        // Test with different sequence length if we have variation data
        if (Size > offset + 2) {
            uint8_t new_seq_len = (Data[offset] % 8) + 1;
            torch::Tensor input2;
            if (batch_first) {
                input2 = torch::randn({batch_size, new_seq_len, input_size});
            } else {
                input2 = torch::randn({new_seq_len, batch_size, input_size});
            }
            
            try {
                auto [output3, hidden3] = lstm->forward(input2, std::make_tuple(h0, c0));
                // output3 is already a tensor from the structured binding
                auto sum3 = output3.sum();
                (void)sum3;
            } catch (...) {
                // Ignore shape mismatch errors
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