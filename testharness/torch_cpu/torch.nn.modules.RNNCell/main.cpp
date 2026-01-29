#include "fuzzer_utils.h"
#include <iostream>

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
        // Need enough bytes for parameters
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for RNNCell first (before creating tensors)
        int64_t input_size = static_cast<int64_t>(Data[offset++] % 64) + 1;  // 1-64
        int64_t hidden_size = static_cast<int64_t>(Data[offset++] % 64) + 1; // 1-64
        int64_t batch_size = static_cast<int64_t>(Data[offset++] % 16) + 1;  // 1-16
        
        // Get nonlinearity type
        bool use_tanh = (Data[offset++] % 2 == 0);
        
        // Get bias flag
        bool bias = (Data[offset++] % 2 == 0);
        
        // Get whether to use initial hidden state
        bool use_hidden = (Data[offset++] % 2 == 0);
        
        // Create RNNCell with options
        torch::nn::RNNCellOptions options(input_size, hidden_size);
        if (use_tanh) {
            options.nonlinearity(torch::kTanh);
        } else {
            options.nonlinearity(torch::kReLU);
        }
        options.bias(bias);
        
        torch::nn::RNNCell rnn_cell(options);
        
        // Create input tensor with correct shape: (batch_size, input_size)
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            // Reshape or recreate to match expected dimensions
            int64_t total_elements = input.numel();
            if (total_elements >= input_size) {
                // Flatten and take what we need
                input = input.flatten().slice(0, 0, batch_size * input_size);
                try {
                    input = input.reshape({batch_size, input_size});
                } catch (...) {
                    input = torch::randn({batch_size, input_size});
                }
            } else {
                input = torch::randn({batch_size, input_size});
            }
        } else {
            input = torch::randn({batch_size, input_size});
        }
        
        // Ensure input is float type
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create hidden state tensor with correct shape: (batch_size, hidden_size)
        torch::Tensor hidden;
        if (use_hidden) {
            if (offset < Size) {
                hidden = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
                int64_t total_elements = hidden.numel();
                if (total_elements >= hidden_size) {
                    hidden = hidden.flatten().slice(0, 0, batch_size * hidden_size);
                    try {
                        hidden = hidden.reshape({batch_size, hidden_size});
                    } catch (...) {
                        hidden = torch::zeros({batch_size, hidden_size});
                    }
                } else {
                    hidden = torch::zeros({batch_size, hidden_size});
                }
            } else {
                hidden = torch::zeros({batch_size, hidden_size});
            }
            
            // Ensure hidden is float type
            if (!hidden.is_floating_point()) {
                hidden = hidden.to(torch::kFloat32);
            }
        }
        
        // Apply RNNCell - forward can take optional hidden state
        torch::Tensor output;
        if (use_hidden) {
            output = rnn_cell->forward(input, hidden);
        } else {
            // When no hidden state provided, RNNCell initializes it to zeros
            output = rnn_cell->forward(input);
        }
        
        // Verify output shape (should be batch_size x hidden_size)
        if (output.dim() != 2 || output.size(0) != batch_size || output.size(1) != hidden_size) {
            std::cerr << "Unexpected output shape: " << output.sizes() << std::endl;
        }
        
        // Additional operations to increase coverage
        // Run multiple steps to simulate sequence processing
        if (offset < Size && (Data[offset] % 4 == 0)) {
            int num_steps = (Data[offset] % 5) + 1;
            torch::Tensor h = output;
            for (int i = 0; i < num_steps; i++) {
                // Create new input for each step
                torch::Tensor step_input = torch::randn({batch_size, input_size});
                h = rnn_cell->forward(step_input, h);
            }
        }
        
        // Test with different batch sizes
        if (offset + 1 < Size && (Data[offset] % 3 == 0)) {
            int64_t new_batch = (Data[offset + 1] % 8) + 1;
            torch::Tensor new_input = torch::randn({new_batch, input_size});
            try {
                torch::Tensor new_output = rnn_cell->forward(new_input);
                (void)new_output;
            } catch (...) {
                // Shape mismatch is expected, silently ignore
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