#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cmath>

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
        // Need at least enough bytes for parameters
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract LSTM parameters from fuzz data
        // Use smaller, bounded values to avoid excessive memory usage
        int64_t input_size = (Data[offset++] % 32) + 1;      // 1-32
        int64_t hidden_size = (Data[offset++] % 32) + 1;     // 1-32
        int64_t num_layers = (Data[offset++] % 3) + 1;       // 1-3
        int64_t seq_len = (Data[offset++] % 16) + 1;         // 1-16
        int64_t batch_size = (Data[offset++] % 8) + 1;       // 1-8
        
        bool bias = Data[offset++] & 1;
        bool batch_first = Data[offset++] & 1;
        bool bidirectional = Data[offset++] & 1;
        
        // Dropout: only apply if num_layers > 1 (PyTorch requirement)
        double dropout = 0.0;
        if (num_layers > 1 && offset < Size) {
            dropout = static_cast<double>(Data[offset++] % 50) / 100.0; // 0.0-0.49
        } else if (offset < Size) {
            offset++; // consume the byte anyway
        }
        
        // Create LSTM module
        torch::nn::LSTM lstm_module(
            torch::nn::LSTMOptions(input_size, hidden_size)
                .num_layers(num_layers)
                .bias(bias)
                .batch_first(batch_first)
                .dropout(dropout)
                .bidirectional(bidirectional)
        );
        
        // Set to eval mode to disable dropout during inference
        lstm_module->eval();
        
        // Create input tensor with correct shape
        // batch_first=true: [batch, seq_len, input_size]
        // batch_first=false: [seq_len, batch, input_size]
        torch::Tensor input_tensor;
        if (batch_first) {
            input_tensor = torch::randn({batch_size, seq_len, input_size});
        } else {
            input_tensor = torch::randn({seq_len, batch_size, input_size});
        }
        
        // Use remaining fuzz data to modify tensor values
        if (offset < Size) {
            size_t remaining = Size - offset;
            auto flat = input_tensor.flatten();
            int64_t num_elements = flat.numel();
            int64_t elements_to_modify = std::min(static_cast<int64_t>(remaining / sizeof(float)), num_elements);
            
            if (elements_to_modify > 0) {
                auto accessor = flat.accessor<float, 1>();
                for (int64_t i = 0; i < elements_to_modify && offset + sizeof(float) <= Size; ++i) {
                    float val;
                    std::memcpy(&val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    // Clamp to reasonable range to avoid NaN/Inf issues
                    if (std::isfinite(val)) {
                        accessor[i] = std::clamp(val, -10.0f, 10.0f);
                    }
                }
            }
        }
        
        // Create initial hidden states
        int64_t num_directions = bidirectional ? 2 : 1;
        torch::Tensor h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
        torch::Tensor c0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
        
        // Test forward pass with initial state
        try {
            auto output = lstm_module->forward(input_tensor, std::make_tuple(h0, c0));
            
            auto output_tensor = std::get<0>(output);
            auto hidden_state = std::get<1>(output);
            auto hn = std::get<0>(hidden_state);
            auto cn = std::get<1>(hidden_state);
            
            // Verify output shapes
            if (batch_first) {
                assert(output_tensor.size(0) == batch_size);
                assert(output_tensor.size(1) == seq_len);
            } else {
                assert(output_tensor.size(0) == seq_len);
                assert(output_tensor.size(1) == batch_size);
            }
            assert(output_tensor.size(2) == hidden_size * num_directions);
            
            // Force computation
            volatile float val1 = output_tensor.sum().item<float>();
            volatile float val2 = hn.sum().item<float>();
            volatile float val3 = cn.sum().item<float>();
            (void)val1; (void)val2; (void)val3;
        } catch (const c10::Error&) {
            // Expected errors from shape mismatches, etc.
        }
        
        // Test forward pass without initial state
        try {
            auto output2 = lstm_module->forward(input_tensor);
            auto output_tensor2 = std::get<0>(output2);
            volatile float val = output_tensor2.sum().item<float>();
            (void)val;
        } catch (const c10::Error&) {
            // Expected errors
        }
        
        // Test with different input (zeros)
        try {
            torch::Tensor zero_input;
            if (batch_first) {
                zero_input = torch::zeros({batch_size, seq_len, input_size});
            } else {
                zero_input = torch::zeros({seq_len, batch_size, input_size});
            }
            auto output3 = lstm_module->forward(zero_input);
            volatile float val = std::get<0>(output3).sum().item<float>();
            (void)val;
        } catch (const c10::Error&) {
            // Expected errors
        }
        
        // Test flatten_parameters (for RNN efficiency)
        try {
            lstm_module->flatten_parameters();
        } catch (const c10::Error&) {
            // May fail in certain configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}