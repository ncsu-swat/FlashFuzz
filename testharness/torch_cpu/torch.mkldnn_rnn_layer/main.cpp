#include "fuzzer_utils.h"
#include <iostream>
#include <vector>

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
        size_t offset = 0;
        
        if (Size < 12) {
            return 0;
        }
        
        // Parse mode (LSTM=0, GRU=1, RNN_TANH=2, RNN_RELU=3)
        uint8_t mode_byte = Data[offset++];
        int64_t mode = mode_byte % 4;
        
        // Parse hidden size (small for performance)
        uint8_t hidden_size_byte = Data[offset++];
        int64_t hidden_size = (hidden_size_byte % 16) + 1; // 1-16
        
        // Parse input size
        uint8_t input_size_byte = Data[offset++];
        int64_t input_size = (input_size_byte % 16) + 1; // 1-16
        
        // Parse sequence length
        uint8_t seq_len_byte = Data[offset++];
        int64_t seq_len = (seq_len_byte % 8) + 1; // 1-8
        
        // Parse batch size
        uint8_t batch_size_byte = Data[offset++];
        int64_t batch_size = (batch_size_byte % 4) + 1; // 1-4
        
        // Parse boolean flags
        uint8_t flags_byte = Data[offset++];
        bool reverse = (flags_byte & 0x01) != 0;
        bool bidirectional = (flags_byte & 0x02) != 0;
        bool batch_first = (flags_byte & 0x04) != 0;
        bool train = (flags_byte & 0x08) != 0;
        bool has_biases = (flags_byte & 0x10) != 0;
        
        // Gates multiplier based on mode
        int64_t gates_multiplier;
        if (mode == 0) { // LSTM
            gates_multiplier = 4;
        } else if (mode == 1) { // GRU
            gates_multiplier = 3;
        } else { // RNN_TANH or RNN_RELU
            gates_multiplier = 1;
        }
        
        // Create input tensor: (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size)
        torch::Tensor input;
        if (batch_first) {
            input = torch::randn({batch_size, seq_len, input_size});
        } else {
            input = torch::randn({seq_len, batch_size, input_size});
        }
        
        // For mkldnn_rnn_layer, weights are for a single layer, single direction
        // weight_ih: (gates_multiplier * hidden_size, input_size)
        // weight_hh: (gates_multiplier * hidden_size, hidden_size)
        torch::Tensor weight_ih = torch::randn({gates_multiplier * hidden_size, input_size});
        torch::Tensor weight_hh = torch::randn({gates_multiplier * hidden_size, hidden_size});
        
        // Biases
        torch::Tensor bias_ih, bias_hh;
        if (has_biases) {
            bias_ih = torch::randn({gates_multiplier * hidden_size});
            bias_hh = torch::randn({gates_multiplier * hidden_size});
        } else {
            bias_ih = torch::zeros({gates_multiplier * hidden_size});
            bias_hh = torch::zeros({gates_multiplier * hidden_size});
        }
        
        // Hidden state: (1, batch_size, hidden_size) for single layer
        torch::Tensor hx = torch::randn({1, batch_size, hidden_size});
        
        // Cell state (only for LSTM)
        torch::Tensor cx;
        if (mode == 0) { // LSTM
            cx = torch::randn({1, batch_size, hidden_size});
        } else {
            cx = torch::zeros({0}); // Empty tensor for non-LSTM
        }
        
        // batch_sizes is empty for non-packed input
        std::vector<int64_t> batch_sizes_vec;
        at::IntArrayRef batch_sizes(batch_sizes_vec);
        
        // mkldnn_rnn_layer is an internal MKLDNN-optimized function
        // It may not be available on all systems (requires MKLDNN/oneDNN support)
        try {
            auto result = torch::mkldnn_rnn_layer(
                input,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                hx,
                cx,
                reverse,
                batch_sizes,
                mode,
                hidden_size,
                /*num_layers=*/1,
                has_biases,
                bidirectional,
                batch_first,
                train
            );
            
            // Access outputs to ensure computation happens
            auto output = std::get<0>(result);
            auto hy = std::get<1>(result);
            auto cy = std::get<2>(result);
            
            // Basic validation
            (void)output.numel();
            (void)hy.numel();
            (void)cy.numel();
        } catch (const c10::Error& e) {
            // MKLDNN may not support all configurations or may not be available
            return 0;
        } catch (const std::runtime_error& e) {
            // MKLDNN not available or unsupported configuration
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}