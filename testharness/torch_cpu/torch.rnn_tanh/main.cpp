#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        // Need enough data to create meaningful inputs
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract configuration from fuzzer data
        uint8_t seq_len_raw = Data[offset++];
        uint8_t batch_size_raw = Data[offset++];
        uint8_t input_size_raw = Data[offset++];
        uint8_t hidden_size_raw = Data[offset++];
        uint8_t config_byte = Data[offset++];
        
        // Derive reasonable dimensions (ensure minimum sizes)
        int64_t seq_len = (seq_len_raw % 10) + 1;        // 1-10
        int64_t batch_size = (batch_size_raw % 8) + 1;   // 1-8
        int64_t input_size = (input_size_raw % 16) + 1;  // 1-16
        int64_t hidden_size = (hidden_size_raw % 16) + 1; // 1-16
        
        // Configuration flags from config_byte
        bool bidirectional = (config_byte & 0x01) != 0;
        bool batch_first = (config_byte & 0x02) != 0;
        bool has_biases = (config_byte & 0x04) != 0;
        bool train = (config_byte & 0x08) != 0;
        
        // For simplicity, use 1 layer (multi-layer requires multiple weight sets)
        int64_t num_layers = 1;
        double dropout = 0.0; // dropout only applies when num_layers > 1
        
        int64_t num_directions = bidirectional ? 2 : 1;
        
        // Create input tensor with correct shape
        torch::Tensor input;
        if (batch_first) {
            input = torch::randn({batch_size, seq_len, input_size});
        } else {
            input = torch::randn({seq_len, batch_size, input_size});
        }
        
        // Use remaining fuzzer data to influence tensor values if available
        if (offset + 4 <= Size) {
            float scale;
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(scale) && std::abs(scale) > 0.001f && std::abs(scale) < 100.0f) {
                input = input * scale;
            }
        }
        
        // Create hidden state: (num_layers * num_directions, batch, hidden_size)
        torch::Tensor h0 = torch::randn({num_layers * num_directions, batch_size, hidden_size});
        
        // Create weight tensors - need one set per direction
        std::vector<torch::Tensor> params;
        
        for (int64_t dir = 0; dir < num_directions; dir++) {
            // weight_ih: (hidden_size, input_size)
            torch::Tensor weight_ih = torch::randn({hidden_size, input_size});
            // weight_hh: (hidden_size, hidden_size)
            torch::Tensor weight_hh = torch::randn({hidden_size, hidden_size});
            
            params.push_back(weight_ih);
            params.push_back(weight_hh);
            
            if (has_biases) {
                // bias_ih: (hidden_size)
                torch::Tensor bias_ih = torch::randn({hidden_size});
                // bias_hh: (hidden_size)
                torch::Tensor bias_hh = torch::randn({hidden_size});
                params.push_back(bias_ih);
                params.push_back(bias_hh);
            }
        }
        
        try {
            // Apply RNN tanh operation
            auto result = torch::rnn_tanh(
                input,
                h0,
                params,
                has_biases,
                num_layers,
                dropout,
                train,
                bidirectional,
                batch_first
            );
            
            // Access the output to ensure computation is performed
            auto output = std::get<0>(result);
            auto h_n = std::get<1>(result);
            
            // Verify output shapes are reasonable
            auto output_sum = output.sum();
            auto h_n_sum = h_n.sum();
            
            // Force computation
            volatile float val = output_sum.item<float>() + h_n_sum.item<float>();
            (void)val;
            
        } catch (const c10::Error& e) {
            // Expected PyTorch errors for invalid configurations
        } catch (const std::runtime_error& e) {
            // Runtime errors from shape mismatches etc.
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}