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
        
        // Ensure input has at least 3 dimensions for LSTM (batch_size, seq_len, input_size)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract dimensions
        int64_t batch_size = input.size(0);
        int64_t seq_len = input.size(1);
        int64_t input_size = input.size(2);
        
        // Create h0 and c0 (initial hidden and cell states)
        int64_t hidden_size = 1 + (offset < Size ? Data[offset++] % 10 : 0);
        int64_t num_layers = 1 + (offset < Size ? Data[offset++] % 3 : 0);
        bool bidirectional = offset < Size ? (Data[offset++] % 2 == 1) : false;
        int64_t directions = bidirectional ? 2 : 1;
        
        // Create h0 and c0 with appropriate dimensions
        torch::Tensor h0 = torch::zeros({num_layers * directions, batch_size, hidden_size});
        torch::Tensor c0 = torch::zeros({num_layers * directions, batch_size, hidden_size});
        
        // Create weight and bias parameters for the LSTM
        // For quantized LSTM, we need to create quantized weights
        double scale = 1.0 / 256.0;
        int64_t zero_point = 0;
        
        // Create weight_ih and weight_hh for each layer
        std::vector<torch::Tensor> weight_ih_l;
        std::vector<torch::Tensor> weight_hh_l;
        std::vector<torch::Tensor> bias_ih_l;
        std::vector<torch::Tensor> bias_hh_l;
        
        for (int64_t layer = 0; layer < num_layers; layer++) {
            for (int64_t direction = 0; direction < directions; direction++) {
                // Weight dimensions: (4*hidden_size, input_size) for first layer, (4*hidden_size, hidden_size*directions) for others
                int64_t layer_input_size = (layer == 0) ? input_size : hidden_size * directions;
                
                // Create quantized weights
                auto w_ih = torch::randn({4 * hidden_size, layer_input_size});
                auto w_hh = torch::randn({4 * hidden_size, hidden_size});
                
                // Quantize the weights
                auto q_w_ih = torch::quantize_per_tensor(w_ih, scale, zero_point, torch::kQInt8);
                auto q_w_hh = torch::quantize_per_tensor(w_hh, scale, zero_point, torch::kQInt8);
                
                weight_ih_l.push_back(q_w_ih);
                weight_hh_l.push_back(q_w_hh);
                
                // Create biases
                auto b_ih = torch::randn({4 * hidden_size});
                auto b_hh = torch::randn({4 * hidden_size});
                
                bias_ih_l.push_back(b_ih);
                bias_hh_l.push_back(b_hh);
            }
        }
        
        // Try different dropout values
        double dropout = offset < Size ? (Data[offset++] % 100) / 100.0 : 0.0;
        
        // Try different has_biases values
        bool has_biases = offset < Size ? (Data[offset++] % 2 == 1) : true;
        
        // Try different batch_first values
        bool batch_first = offset < Size ? (Data[offset++] % 2 == 1) : false;
        
        // If batch_first is true, permute the input tensor
        if (batch_first) {
            input = input.permute({1, 0, 2});
        }
        
        try {
            // Call quantized_lstm using at::quantized_lstm
            auto result = at::quantized_lstm(
                input,
                std::make_tuple(h0, c0),
                weight_ih_l,
                weight_hh_l,
                bias_ih_l,
                bias_hh_l,
                has_biases,
                num_layers,
                dropout,
                false, // train
                bidirectional,
                batch_first
            );
            
            // Unpack the result
            auto output = std::get<0>(result);
            auto hidden = std::get<1>(result);
            
            // Verify output shape
            if (output.dim() != 3) {
                throw std::runtime_error("Output dimension mismatch");
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and handled
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}