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
        // Need at least some data to configure the LSTM cell
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract configuration from fuzzer data
        uint8_t input_size_byte = Data[offset++];
        uint8_t hidden_size_byte = Data[offset++];
        uint8_t batch_size_byte = Data[offset++];
        uint8_t config_byte = Data[offset++];

        // Constrain sizes to reasonable ranges to avoid OOM
        int64_t input_size = (input_size_byte % 32) + 1;   // 1-32
        int64_t hidden_size = (hidden_size_byte % 32) + 1; // 1-32
        int64_t batch_size = (batch_size_byte % 8) + 1;    // 1-8
        bool use_bias = (config_byte & 0x01) != 0;
        bool provide_hidden = (config_byte & 0x02) != 0;

        // Create LSTM cell with extracted parameters
        torch::nn::LSTMCellOptions options(input_size, hidden_size);
        options.bias(use_bias);
        torch::nn::LSTMCell lstm_cell(options);

        // Create input tensor with correct shape: (batch_size, input_size)
        torch::Tensor input;
        if (offset < Size) {
            torch::Tensor raw_input = fuzzer_utils::createTensor(Data, Size, offset);
            // Reshape or create a properly sized input
            int64_t total_elements = raw_input.numel();
            if (total_elements >= batch_size * input_size) {
                input = raw_input.flatten().slice(0, 0, batch_size * input_size)
                            .reshape({batch_size, input_size}).to(torch::kFloat);
            } else if (total_elements > 0) {
                // Pad with zeros if not enough elements
                auto flat = raw_input.flatten().to(torch::kFloat);
                auto padding = torch::zeros({batch_size * input_size - total_elements});
                input = torch::cat({flat, padding}).reshape({batch_size, input_size});
            } else {
                input = torch::randn({batch_size, input_size});
            }
        } else {
            input = torch::randn({batch_size, input_size});
        }

        // Ensure input is float
        input = input.to(torch::kFloat);

        // Optionally provide hidden states
        std::tuple<torch::Tensor, torch::Tensor> hx;
        
        if (provide_hidden) {
            torch::Tensor h0, c0;
            
            // Create h0
            if (offset < Size) {
                torch::Tensor raw_h0 = fuzzer_utils::createTensor(Data, Size, offset);
                int64_t h_elements = raw_h0.numel();
                if (h_elements >= batch_size * hidden_size) {
                    h0 = raw_h0.flatten().slice(0, 0, batch_size * hidden_size)
                             .reshape({batch_size, hidden_size}).to(torch::kFloat);
                } else {
                    h0 = torch::zeros({batch_size, hidden_size});
                }
            } else {
                h0 = torch::zeros({batch_size, hidden_size});
            }

            // Create c0
            if (offset < Size) {
                torch::Tensor raw_c0 = fuzzer_utils::createTensor(Data, Size, offset);
                int64_t c_elements = raw_c0.numel();
                if (c_elements >= batch_size * hidden_size) {
                    c0 = raw_c0.flatten().slice(0, 0, batch_size * hidden_size)
                             .reshape({batch_size, hidden_size}).to(torch::kFloat);
                } else {
                    c0 = torch::zeros({batch_size, hidden_size});
                }
            } else {
                c0 = torch::zeros({batch_size, hidden_size});
            }

            hx = std::make_tuple(h0, c0);
            
            // Call with hidden state
            auto result = lstm_cell(input, hx);
            auto h1 = std::get<0>(result);
            auto c1 = std::get<1>(result);
            
            // Use the results
            auto sum_val = torch::sum(h1).item<float>() + torch::sum(c1).item<float>();
            (void)sum_val;
        } else {
            // Call without hidden state (will use zeros internally)
            auto result = lstm_cell(input);
            auto h1 = std::get<0>(result);
            auto c1 = std::get<1>(result);
            
            // Use the results
            auto sum_val = torch::sum(h1).item<float>() + torch::sum(c1).item<float>();
            (void)sum_val;
        }

        // Test multiple forward passes to explore state-dependent behavior
        if ((config_byte & 0x04) != 0) {
            torch::Tensor h_state = torch::zeros({batch_size, hidden_size});
            torch::Tensor c_state = torch::zeros({batch_size, hidden_size});
            
            int num_steps = ((config_byte >> 4) % 4) + 1; // 1-4 steps
            for (int i = 0; i < num_steps; i++) {
                auto result = lstm_cell(input, std::make_tuple(h_state, c_state));
                h_state = std::get<0>(result);
                c_state = std::get<1>(result);
            }
            
            auto final_sum = torch::sum(h_state).item<float>();
            (void)final_sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}