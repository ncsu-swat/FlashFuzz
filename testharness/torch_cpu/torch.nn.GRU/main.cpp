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
        if (Size < 12) return 0;  // Need minimum data for meaningful test
        
        size_t offset = 0;
        
        // Parse GRU parameters from the beginning of data
        int64_t input_size = (Data[offset++] % 16) + 1;    // 1-16
        int64_t hidden_size = (Data[offset++] % 16) + 1;   // 1-16
        int64_t num_layers = (Data[offset++] % 3) + 1;     // 1-3
        bool bias = Data[offset++] % 2;                    // 0-1
        bool batch_first = Data[offset++] % 2;             // 0-1
        float dropout_val = (num_layers > 1) ? static_cast<float>(Data[offset++] % 50) / 100.0f : 0.0f; // 0.0-0.49, only if num_layers > 1
        if (num_layers == 1) offset++;  // Consume the byte anyway
        bool bidirectional = Data[offset++] % 2;           // 0-1
        
        // Parse sequence dimensions
        int64_t seq_len = (Data[offset++] % 8) + 1;        // 1-8
        int64_t batch_size = (Data[offset++] % 4) + 1;     // 1-4
        
        // Determine if we should provide initial hidden state
        bool provide_h0 = Data[offset++] % 2;
        
        // Create GRU module
        torch::nn::GRU gru(torch::nn::GRUOptions(input_size, hidden_size)
                            .num_layers(num_layers)
                            .bias(bias)
                            .batch_first(batch_first)
                            .dropout(dropout_val)
                            .bidirectional(bidirectional));
        
        // Put module in eval mode to disable dropout randomness issues
        gru->eval();
        
        int64_t num_directions = bidirectional ? 2 : 1;
        
        // Create input tensor with correct shape
        // GRU expects: (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first
        torch::Tensor input;
        if (batch_first) {
            input = torch::randn({batch_size, seq_len, input_size});
        } else {
            input = torch::randn({seq_len, batch_size, input_size});
        }
        
        // Use remaining fuzzer data to perturb the input
        if (offset < Size) {
            torch::Tensor fuzz_data = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                // Try to use fuzzer data to modify input values
                auto flat_input = input.flatten();
                auto flat_fuzz = fuzz_data.flatten().to(flat_input.dtype());
                int64_t copy_len = std::min(flat_input.size(0), flat_fuzz.size(0));
                if (copy_len > 0) {
                    flat_input.slice(0, 0, copy_len).copy_(flat_fuzz.slice(0, 0, copy_len));
                }
                input = flat_input.reshape(input.sizes());
            } catch (...) {
                // Ignore reshape/copy errors, use random input
            }
        }
        
        // Inner try-catch for expected runtime errors (shape mismatches, etc.)
        try {
            if (provide_h0) {
                // Create initial hidden state: (num_layers * num_directions, batch, hidden_size)
                torch::Tensor h0 = torch::randn({num_layers * num_directions, batch_size, hidden_size});
                
                // Perturb h0 with fuzzer data if available
                if (offset < Size) {
                    torch::Tensor fuzz_h0 = fuzzer_utils::createTensor(Data, Size, offset);
                    try {
                        auto flat_h0 = h0.flatten();
                        auto flat_fuzz = fuzz_h0.flatten().to(flat_h0.dtype());
                        int64_t copy_len = std::min(flat_h0.size(0), flat_fuzz.size(0));
                        if (copy_len > 0) {
                            flat_h0.slice(0, 0, copy_len).copy_(flat_fuzz.slice(0, 0, copy_len));
                        }
                        h0 = flat_h0.reshape(h0.sizes());
                    } catch (...) {
                        // Ignore, use random h0
                    }
                }
                
                // Forward pass with hidden state
                auto output = gru->forward(input, h0);
                auto output_tensor = std::get<0>(output);
                auto hidden_state = std::get<1>(output);
                
                // Use the outputs
                auto sum = output_tensor.sum() + hidden_state.sum();
                (void)sum;
            } else {
                // Forward pass without initial hidden state (uses zeros)
                auto output = gru->forward(input);
                auto output_tensor = std::get<0>(output);
                auto hidden_state = std::get<1>(output);
                
                // Use the outputs
                auto sum = output_tensor.sum() + hidden_state.sum();
                (void)sum;
            }
        } catch (...) {
            // Silently catch expected errors like shape mismatches
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