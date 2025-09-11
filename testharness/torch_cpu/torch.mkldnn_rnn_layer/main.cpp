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
        
        // Parse mode (LSTM, GRU, RNN_TANH, RNN_RELU)
        uint8_t mode_byte = (offset < Size) ? Data[offset++] : 0;
        int64_t mode = mode_byte % 4; // 0: LSTM, 1: GRU, 2: RNN_TANH, 3: RNN_RELU
        
        // Parse direction (unidirectional or bidirectional)
        uint8_t direction_byte = (offset < Size) ? Data[offset++] : 0;
        bool bidirectional = direction_byte % 2 == 1;
        
        // Parse number of layers
        uint8_t num_layers_byte = (offset < Size) ? Data[offset++] : 0;
        int64_t num_layers = (num_layers_byte % 3) + 1; // 1-3 layers
        
        // Parse hidden size
        uint8_t hidden_size_byte = (offset < Size) ? Data[offset++] : 0;
        int64_t hidden_size = (hidden_size_byte % 32) + 1; // 1-32 hidden size
        
        // Parse input size
        int64_t input_size = 0;
        if (input.dim() > 0) {
            input_size = input.size(-1);
        } else {
            uint8_t input_size_byte = (offset < Size) ? Data[offset++] : 0;
            input_size = (input_size_byte % 32) + 1; // 1-32 input size
            
            // Create a default input tensor if the one we created is a scalar
            input = torch::randn({1, 1, input_size});
        }
        
        // Create weight and bias tensors
        torch::Tensor weight_ih, weight_hh, bias_ih, bias_hh;
        
        // Create weights based on mode
        int64_t gates_multiplier = 1;
        if (mode == 0) { // LSTM
            gates_multiplier = 4;
        } else if (mode == 1) { // GRU
            gates_multiplier = 3;
        } else { // RNN
            gates_multiplier = 1;
        }
        
        int64_t dir_multiplier = bidirectional ? 2 : 1;
        
        // Create weights and biases
        weight_ih = torch::randn({dir_multiplier * num_layers, gates_multiplier * hidden_size, input_size});
        weight_hh = torch::randn({dir_multiplier * num_layers, gates_multiplier * hidden_size, hidden_size});
        bias_ih = torch::randn({dir_multiplier * num_layers, gates_multiplier * hidden_size});
        bias_hh = torch::randn({dir_multiplier * num_layers, gates_multiplier * hidden_size});
        
        // Parse batch_first
        uint8_t batch_first_byte = (offset < Size) ? Data[offset++] : 0;
        bool batch_first = batch_first_byte % 2 == 1;
        
        // Parse training mode
        uint8_t train_byte = (offset < Size) ? Data[offset++] : 0;
        bool train = train_byte % 2 == 1;
        
        // Parse reverse
        uint8_t reverse_byte = (offset < Size) ? Data[offset++] : 0;
        bool reverse = reverse_byte % 2 == 1;
        
        // Create initial hidden and cell states if needed
        torch::Tensor h0, c0;
        int64_t batch_size = input.size(batch_first ? 0 : 1);
        h0 = torch::randn({dir_multiplier * num_layers, batch_size, hidden_size});
        
        if (mode == 0) { // LSTM needs both h0 and c0
            c0 = torch::randn({dir_multiplier * num_layers, batch_size, hidden_size});
        } else { // GRU and RNN only need h0
            c0 = torch::empty({0});
        }
        
        // Create batch_sizes (for packed sequences, use simple case)
        std::vector<int64_t> batch_sizes_vec = {batch_size};
        at::IntArrayRef batch_sizes(batch_sizes_vec);
        
        // Call mkldnn_rnn_layer
        try {
            auto result = torch::mkldnn_rnn_layer(
                input, weight_ih, weight_hh, bias_ih, bias_hh,
                h0, c0, reverse, batch_sizes, mode, hidden_size, num_layers, 
                true, bidirectional, batch_first, train
            );
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
