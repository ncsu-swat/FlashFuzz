#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Need enough data for parameters
        if (Size < 16) {
            return 0;
        }
        
        // Extract RNN parameters from fuzzer data
        int64_t seq_len = (Data[offset++] % 10) + 1;      // 1-10
        int64_t batch_size = (Data[offset++] % 8) + 1;    // 1-8
        int64_t input_size = (Data[offset++] % 16) + 1;   // 1-16
        int64_t hidden_size = (Data[offset++] % 16) + 1;  // 1-16
        int64_t num_layers = (Data[offset++] % 3) + 1;    // 1-3
        
        bool batch_first = Data[offset++] & 0x1;
        bool bidirectional = Data[offset++] & 0x1;
        bool has_h0 = Data[offset++] & 0x1;
        
        double dropout = 0.0;
        if (num_layers > 1 && offset < Size) {
            dropout = (Data[offset++] % 50) / 100.0;  // 0.0 - 0.49
        }
        
        int64_t num_directions = bidirectional ? 2 : 1;
        
        // Create input tensor with proper shape
        torch::Tensor input;
        if (batch_first) {
            input = torch::randn({batch_size, seq_len, input_size});
        } else {
            input = torch::randn({seq_len, batch_size, input_size});
        }
        
        // Create initial hidden state if requested
        torch::Tensor h0;
        if (has_h0) {
            h0 = torch::randn({num_layers * num_directions, batch_size, hidden_size});
        }
        
        // Variant 1: Test torch::nn::RNN with ReLU nonlinearity
        try {
            auto rnn_options = torch::nn::RNNOptions(input_size, hidden_size)
                .num_layers(num_layers)
                .batch_first(batch_first)
                .bidirectional(bidirectional)
                .nonlinearity(torch::kReLU);
            
            if (num_layers > 1) {
                rnn_options.dropout(dropout);
            }
            
            auto rnn = torch::nn::RNN(rnn_options);
            
            if (has_h0) {
                auto [output, hn] = rnn->forward(input, h0);
                (void)output;
                (void)hn;
            } else {
                auto [output, hn] = rnn->forward(input);
                (void)output;
                (void)hn;
            }
        } catch (const std::exception&) {
            // Shape mismatches or other expected errors
        }
        
        // Variant 2: Test torch::nn::RNNCell with ReLU nonlinearity
        try {
            auto rnn_cell = torch::nn::RNNCell(
                torch::nn::RNNCellOptions(input_size, hidden_size)
                    .nonlinearity(torch::kReLU));
            
            // Create single timestep input: (batch, input_size)
            torch::Tensor cell_input = torch::randn({batch_size, input_size});
            torch::Tensor hx = torch::randn({batch_size, hidden_size});
            
            // Forward with hidden state
            torch::Tensor h_new = rnn_cell->forward(cell_input, hx);
            (void)h_new;
            
            // Forward without explicit hidden state (uses zeros)
            torch::Tensor h_new2 = rnn_cell->forward(cell_input);
            (void)h_new2;
        } catch (const std::exception&) {
            // Expected errors
        }
        
        // Variant 3: Test the low-level torch::rnn_relu function if available
        try {
            // Create properly shaped weight tensors for a single layer
            // weight_ih: (hidden_size, input_size)
            // weight_hh: (hidden_size, hidden_size)
            // bias_ih: (hidden_size)
            // bias_hh: (hidden_size)
            
            torch::Tensor weight_ih = torch::randn({hidden_size, input_size});
            torch::Tensor weight_hh = torch::randn({hidden_size, hidden_size});
            torch::Tensor bias_ih = torch::randn({hidden_size});
            torch::Tensor bias_hh = torch::randn({hidden_size});
            
            // For multi-layer, we need weights for each layer
            std::vector<torch::Tensor> all_weights;
            for (int64_t layer = 0; layer < num_layers; ++layer) {
                int64_t layer_input_size = (layer == 0) ? input_size : hidden_size * num_directions;
                
                for (int64_t dir = 0; dir < num_directions; ++dir) {
                    all_weights.push_back(torch::randn({hidden_size, layer_input_size}));  // weight_ih
                    all_weights.push_back(torch::randn({hidden_size, hidden_size}));       // weight_hh
                    all_weights.push_back(torch::randn({hidden_size}));                    // bias_ih
                    all_weights.push_back(torch::randn({hidden_size}));                    // bias_hh
                }
            }
            
            torch::Tensor hx_func = torch::randn({num_layers * num_directions, batch_size, hidden_size});
            
            // Prepare input in non-batch-first format for the functional API
            torch::Tensor func_input = batch_first ? input.transpose(0, 1) : input;
            
            auto result = torch::rnn_relu(
                func_input,
                hx_func,
                all_weights,
                true,           // has_biases
                num_layers,
                dropout,
                false,          // training = false for inference
                bidirectional,
                false           // batch_first = false (we transposed above)
            );
            
            (void)std::get<0>(result);  // output
            (void)std::get<1>(result);  // hn
        } catch (const std::exception&) {
            // Expected errors from parameter mismatches
        }
        
        // Variant 4: Test without biases
        try {
            std::vector<torch::Tensor> weights_no_bias;
            for (int64_t layer = 0; layer < num_layers; ++layer) {
                int64_t layer_input_size = (layer == 0) ? input_size : hidden_size * num_directions;
                
                for (int64_t dir = 0; dir < num_directions; ++dir) {
                    weights_no_bias.push_back(torch::randn({hidden_size, layer_input_size}));
                    weights_no_bias.push_back(torch::randn({hidden_size, hidden_size}));
                }
            }
            
            torch::Tensor hx_nb = torch::randn({num_layers * num_directions, batch_size, hidden_size});
            torch::Tensor func_input = batch_first ? input.transpose(0, 1) : input;
            
            auto result = torch::rnn_relu(
                func_input,
                hx_nb,
                weights_no_bias,
                false,          // has_biases = false
                num_layers,
                dropout,
                false,          // training
                bidirectional,
                false           // batch_first
            );
            
            (void)std::get<0>(result);
            (void)std::get<1>(result);
        } catch (const std::exception&) {
            // Expected errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}