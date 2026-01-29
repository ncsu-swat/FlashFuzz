#include "fuzzer_utils.h"
#include <iostream>

// --- Fuzzer Entry Point ---
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
        
        if (Size < 16) return 0; // Need minimum data for parameters
        
        // Parse RNN parameters from the data
        // Parse mode (RNN=0, LSTM=1, GRU=2)
        uint8_t mode_byte = Data[offset++] % 3;
        int64_t mode = static_cast<int64_t>(mode_byte);
        
        // Parse dimensions - keep small for reasonable memory usage
        int64_t seq_length = static_cast<int64_t>((Data[offset++] % 8) + 1);      // 1-8
        int64_t batch_size_val = static_cast<int64_t>((Data[offset++] % 4) + 1);  // 1-4
        int64_t input_size = static_cast<int64_t>((Data[offset++] % 16) + 1);     // 1-16
        int64_t hidden_size = static_cast<int64_t>((Data[offset++] % 16) + 1);    // 1-16
        int64_t num_layers = static_cast<int64_t>((Data[offset++] % 3) + 1);      // 1-3
        
        // Parse boolean flags
        bool batch_first = (Data[offset++] % 2) == 1;
        bool bidirectional = (Data[offset++] % 2) == 1;
        bool train = (Data[offset++] % 2) == 1;
        
        // Parse dropout (0.0 to 0.5 range)
        double dropout = static_cast<double>(Data[offset++] % 128) / 255.0;
        
        // Compute derived values
        int64_t num_directions = bidirectional ? 2 : 1;
        
        // Weight stride
        int64_t weight_stride0 = static_cast<int64_t>((Data[offset++] % 4) + 1);
        
        // Create input tensor: (seq_length, batch_size, input_size) or (batch_size, seq_length, input_size)
        torch::Tensor input;
        if (batch_first) {
            input = torch::randn({batch_size_val, seq_length, input_size});
        } else {
            input = torch::randn({seq_length, batch_size_val, input_size});
        }
        
        // Create weight tensors
        // Weight dimensions depend on mode:
        // RNN: gate_size = hidden_size
        // GRU: gate_size = 3 * hidden_size  
        // LSTM: gate_size = 4 * hidden_size
        int64_t gate_multiplier = 1;
        if (mode == 1) gate_multiplier = 4;  // LSTM
        else if (mode == 2) gate_multiplier = 3;  // GRU
        
        std::vector<torch::Tensor> weight_list;
        
        // Create weight tensors for each layer and direction
        for (int64_t layer = 0; layer < num_layers; ++layer) {
            for (int64_t dir = 0; dir < num_directions; ++dir) {
                int64_t layer_input_size = (layer == 0) ? input_size : hidden_size * num_directions;
                
                // weight_ih: (gate_size, layer_input_size)
                torch::Tensor weight_ih = torch::randn({gate_multiplier * hidden_size, layer_input_size});
                weight_list.push_back(weight_ih);
                
                // weight_hh: (gate_size, hidden_size)
                torch::Tensor weight_hh = torch::randn({gate_multiplier * hidden_size, hidden_size});
                weight_list.push_back(weight_hh);
                
                // bias_ih: (gate_size,)
                torch::Tensor bias_ih = torch::randn({gate_multiplier * hidden_size});
                weight_list.push_back(bias_ih);
                
                // bias_hh: (gate_size,)
                torch::Tensor bias_hh = torch::randn({gate_multiplier * hidden_size});
                weight_list.push_back(bias_hh);
            }
        }
        
        // Create hidden state tensor: (num_layers * num_directions, batch_size, hidden_size)
        torch::Tensor hx = torch::randn({num_layers * num_directions, batch_size_val, hidden_size});
        
        // Create cell state tensor for LSTM
        std::optional<torch::Tensor> cx = std::nullopt;
        if (mode == 1) { // LSTM needs cell state
            cx = torch::randn({num_layers * num_directions, batch_size_val, hidden_size});
        }
        
        // Batch sizes (for packed sequences - use uniform batch sizes here)
        std::vector<int64_t> batch_sizes;
        for (int64_t i = 0; i < seq_length; ++i) {
            batch_sizes.push_back(batch_size_val);
        }
        
        // Dropout state (optional)
        std::optional<torch::Tensor> dropout_state = std::nullopt;
        
        // Try to call miopen_rnn
        // Note: This function requires MIOpen backend (AMD GPU)
        // On CPU builds, this will throw an error
        try {
            auto result = torch::miopen_rnn(
                input,
                weight_list,
                weight_stride0,
                hx,
                cx,
                mode,
                hidden_size,
                num_layers,
                batch_first,
                dropout,
                train,
                bidirectional,
                batch_sizes,
                dropout_state
            );
            
            // Access results to ensure computation happened
            auto output = std::get<0>(result);
            auto hy = std::get<1>(result);
            auto cy = std::get<2>(result);
            auto reserve = std::get<3>(result);
            auto weight_buf = std::get<4>(result);
            
            (void)output.numel();
            (void)hy.numel();
        } catch (const c10::Error& e) {
            // Expected errors: MIOpen not available, wrong backend, shape mismatches
            // These are expected on CPU builds
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}