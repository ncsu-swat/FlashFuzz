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
        // Need sufficient data for LSTM parameters
        if (Size < 16) {
            return 0;
        }

        size_t offset = 0;

        // Extract configuration from fuzzer data
        int64_t seq_len = (Data[offset++] % 8) + 1;      // 1-8
        int64_t batch_size = (Data[offset++] % 4) + 1;   // 1-4
        int64_t input_size = (Data[offset++] % 8) + 1;   // 1-8
        int64_t hidden_size = (Data[offset++] % 8) + 1;  // 1-8
        int64_t num_layers = (Data[offset++] % 2) + 1;   // 1-2
        bool has_biases = (Data[offset++] % 2 == 0);
        bool batch_first = (Data[offset++] % 2 == 0);
        bool bidirectional = (Data[offset++] % 2 == 0);
        double dropout = 0.0; // Must be 0 for single layer or eval mode

        int64_t num_directions = bidirectional ? 2 : 1;

        // Create input tensor with proper shape
        // Shape: (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size) if batch_first
        torch::Tensor input;
        if (batch_first) {
            input = torch::randn({batch_size, seq_len, input_size});
        } else {
            input = torch::randn({seq_len, batch_size, input_size});
        }

        // Create hidden state tensors
        // h0, c0 shape: (num_layers * num_directions, batch_size, hidden_size)
        torch::Tensor h0 = torch::randn({num_layers * num_directions, batch_size, hidden_size});
        torch::Tensor c0 = torch::randn({num_layers * num_directions, batch_size, hidden_size});

        // Build weight tensors for all layers
        // For each layer: weight_ih, weight_hh, [bias_ih, bias_hh]
        std::vector<torch::Tensor> params;

        for (int64_t layer = 0; layer < num_layers; ++layer) {
            for (int64_t direction = 0; direction < num_directions; ++direction) {
                int64_t layer_input_size = (layer == 0) ? input_size : hidden_size * num_directions;

                // weight_ih: (4 * hidden_size, layer_input_size)
                torch::Tensor weight_ih = torch::randn({4 * hidden_size, layer_input_size});
                // weight_hh: (4 * hidden_size, hidden_size)
                torch::Tensor weight_hh = torch::randn({4 * hidden_size, hidden_size});

                params.push_back(weight_ih);
                params.push_back(weight_hh);

                if (has_biases) {
                    // bias_ih, bias_hh: (4 * hidden_size,)
                    torch::Tensor bias_ih = torch::randn({4 * hidden_size});
                    torch::Tensor bias_hh = torch::randn({4 * hidden_size});
                    params.push_back(bias_ih);
                    params.push_back(bias_hh);
                }
            }
        }

        try {
            // Call torch::lstm
            // Signature: lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)
            std::tuple<torch::Tensor, torch::Tensor> hx = std::make_tuple(h0, c0);

            auto result = torch::lstm(
                input,
                {h0, c0},           // hx as TensorList
                params,              // params as TensorList
                has_biases,
                num_layers,
                dropout,
                false,               // train=false (no dropout in eval)
                bidirectional,
                batch_first
            );

            // Unpack results to ensure they're computed
            auto output = std::get<0>(result);
            auto hn = std::get<1>(result);
            auto cn = std::get<2>(result);

            // Verify output shapes are reasonable
            (void)output.size(0);
            (void)hn.size(0);
            (void)cn.size(0);

        } catch (const c10::Error& e) {
            // Expected PyTorch errors (shape mismatch, etc.)
            return 0;
        } catch (const std::runtime_error& e) {
            // Runtime errors from invalid configurations
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