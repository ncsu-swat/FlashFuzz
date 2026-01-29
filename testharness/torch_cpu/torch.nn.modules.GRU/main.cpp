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
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }
        
        // Parse GRU parameters first (before creating tensors)
        uint8_t input_size_byte = Data[offset++];
        uint8_t hidden_size_byte = Data[offset++];
        uint8_t num_layers_byte = Data[offset++];
        uint8_t flags_byte = Data[offset++];
        uint8_t seq_len_byte = Data[offset++];
        uint8_t batch_size_byte = Data[offset++];
        
        // Convert to usable parameters with reasonable bounds
        int64_t input_size = (input_size_byte % 32) + 1;    // 1-32
        int64_t hidden_size = (hidden_size_byte % 32) + 1;  // 1-32
        int64_t num_layers = (num_layers_byte % 3) + 1;     // 1-3
        int64_t seq_len = (seq_len_byte % 16) + 1;          // 1-16
        int64_t batch_size = (batch_size_byte % 8) + 1;     // 1-8
        
        // Extract boolean flags
        bool batch_first = (flags_byte & 0x01) != 0;
        bool bidirectional = (flags_byte & 0x02) != 0;
        bool use_h0 = (flags_byte & 0x04) != 0;
        
        // Dropout only applies when num_layers > 1
        double dropout = 0.0;
        if (num_layers > 1 && Size - offset >= 1) {
            uint8_t dropout_byte = Data[offset++];
            dropout = static_cast<double>(dropout_byte % 50) / 100.0;  // 0.0-0.49
        }
        
        int64_t num_directions = bidirectional ? 2 : 1;
        
        // Create GRU module
        torch::nn::GRU gru(
            torch::nn::GRUOptions(input_size, hidden_size)
                .num_layers(num_layers)
                .batch_first(batch_first)
                .bidirectional(bidirectional)
                .dropout(dropout)
        );
        gru->eval();  // Set to eval mode to avoid dropout randomness issues
        
        // Create input tensor with correct shape
        // GRU expects: (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first
        torch::Tensor input;
        if (batch_first) {
            input = torch::randn({batch_size, seq_len, input_size});
        } else {
            input = torch::randn({seq_len, batch_size, input_size});
        }
        
        // Use fuzzer data to perturb the input if available
        if (Size - offset >= 4) {
            float scale = static_cast<float>(Data[offset++]) / 25.5f;  // 0-10 scale
            float bias = static_cast<float>(Data[offset++]) / 25.5f - 5.0f;  // -5 to 5 bias
            input = input * scale + bias;
        }
        
        // Create initial hidden state if requested
        torch::Tensor h0;
        if (use_h0) {
            // h0 shape: (num_layers * num_directions, batch, hidden_size)
            h0 = torch::randn({num_layers * num_directions, batch_size, hidden_size});
            
            // Perturb h0 with fuzzer data
            if (Size - offset >= 2) {
                float h0_scale = static_cast<float>(Data[offset++]) / 25.5f;
                h0 = h0 * h0_scale;
            }
        }
        
        // Apply GRU
        std::tuple<torch::Tensor, torch::Tensor> output;
        try {
            if (use_h0) {
                output = gru->forward(input, h0);
            } else {
                output = gru->forward(input);
            }
        } catch (const c10::Error &e) {
            // Expected errors from shape mismatches, etc.
            return 0;
        }
        
        // Access output tensors
        torch::Tensor output_seq = std::get<0>(output);
        torch::Tensor h_n = std::get<1>(output);
        
        // Verify output shapes
        // output_seq: (seq_len, batch, num_directions * hidden_size) or (batch, seq_len, ...) if batch_first
        // h_n: (num_layers * num_directions, batch, hidden_size)
        
        // Perform operations on output to ensure computation happens
        auto sum1 = output_seq.sum();
        auto sum2 = h_n.sum();
        auto total_sum = sum1 + sum2;
        
        // Additional operations to increase coverage
        auto mean_out = output_seq.mean();
        auto max_out = output_seq.max();
        
        // Test backward pass (gradients)
        if ((flags_byte & 0x08) != 0) {
            try {
                auto input_grad = input.clone().detach().requires_grad_(true);
                torch::Tensor h0_grad;
                
                std::tuple<torch::Tensor, torch::Tensor> grad_output;
                if (use_h0) {
                    h0_grad = h0.clone().detach().requires_grad_(true);
                    grad_output = gru->forward(input_grad, h0_grad);
                } else {
                    grad_output = gru->forward(input_grad);
                }
                
                auto loss = std::get<0>(grad_output).sum();
                loss.backward();
            } catch (const c10::Error &e) {
                // Gradient computation may fail for some configurations
            }
        }
        
        // Prevent compiler from optimizing away
        if (total_sum.item<float>() == -12345.6789f) {
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