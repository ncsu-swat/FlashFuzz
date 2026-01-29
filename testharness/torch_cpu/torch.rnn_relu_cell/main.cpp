#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need enough data for dimensions and some tensor data
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract dimensions from fuzz data for controlled tensor creation
        int64_t batch_size = 1 + (Data[offset++] % 16);      // 1-16
        int64_t input_size = 1 + (Data[offset++] % 32);      // 1-32
        int64_t hidden_size = 1 + (Data[offset++] % 32);     // 1-32
        bool use_bias = Data[offset++] % 2 == 0;

        // Determine dtype from fuzz data
        auto dtype_options = torch::TensorOptions().dtype(torch::kFloat32);

        // Create input tensor: (batch, input_size)
        torch::Tensor input;
        if (offset + batch_size * input_size * sizeof(float) <= Size) {
            input = torch::from_blob(
                (void*)(Data + offset),
                {batch_size, input_size},
                dtype_options
            ).clone();
            offset += batch_size * input_size * sizeof(float);
        } else {
            input = torch::zeros({batch_size, input_size}, dtype_options);
            // Fill with available data
            auto input_accessor = input.accessor<float, 2>();
            for (int64_t i = 0; i < batch_size && offset < Size; i++) {
                for (int64_t j = 0; j < input_size && offset < Size; j++) {
                    input_accessor[i][j] = static_cast<float>(Data[offset++]) / 128.0f - 1.0f;
                }
            }
        }

        // Create hidden state: (batch, hidden_size)
        torch::Tensor hx = torch::zeros({batch_size, hidden_size}, dtype_options);
        {
            auto hx_accessor = hx.accessor<float, 2>();
            for (int64_t i = 0; i < batch_size && offset < Size; i++) {
                for (int64_t j = 0; j < hidden_size && offset < Size; j++) {
                    hx_accessor[i][j] = static_cast<float>(Data[offset++]) / 128.0f - 1.0f;
                }
            }
        }

        // Create weight_ih: (hidden_size, input_size)
        torch::Tensor w_ih = torch::zeros({hidden_size, input_size}, dtype_options);
        {
            auto w_accessor = w_ih.accessor<float, 2>();
            for (int64_t i = 0; i < hidden_size && offset < Size; i++) {
                for (int64_t j = 0; j < input_size && offset < Size; j++) {
                    w_accessor[i][j] = static_cast<float>(Data[offset++]) / 128.0f - 1.0f;
                }
            }
        }

        // Create weight_hh: (hidden_size, hidden_size)
        torch::Tensor w_hh = torch::zeros({hidden_size, hidden_size}, dtype_options);
        {
            auto w_accessor = w_hh.accessor<float, 2>();
            for (int64_t i = 0; i < hidden_size && offset < Size; i++) {
                for (int64_t j = 0; j < hidden_size && offset < Size; j++) {
                    w_accessor[i][j] = static_cast<float>(Data[offset++]) / 128.0f - 1.0f;
                }
            }
        }

        // Create bias tensors if needed
        torch::Tensor b_ih, b_hh;
        if (use_bias) {
            b_ih = torch::zeros({hidden_size}, dtype_options);
            b_hh = torch::zeros({hidden_size}, dtype_options);
            
            auto b_ih_accessor = b_ih.accessor<float, 1>();
            for (int64_t i = 0; i < hidden_size && offset < Size; i++) {
                b_ih_accessor[i] = static_cast<float>(Data[offset++]) / 128.0f - 1.0f;
            }
            
            auto b_hh_accessor = b_hh.accessor<float, 1>();
            for (int64_t i = 0; i < hidden_size && offset < Size; i++) {
                b_hh_accessor[i] = static_cast<float>(Data[offset++]) / 128.0f - 1.0f;
            }
        }

        // Apply the RNN ReLU cell operation
        torch::Tensor output;
        
        try {
            if (use_bias) {
                output = torch::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
            } else {
                output = torch::rnn_relu_cell(input, hx, w_ih, w_hh);
            }

            // Verify output shape: should be (batch, hidden_size)
            if (output.dim() == 2 && output.size(0) == batch_size && output.size(1) == hidden_size) {
                // Perform operations on output to ensure it's computed
                auto sum = output.sum();
                auto mean = output.mean();
                auto max_val = output.max();
                
                // Force computation
                (void)sum.item<float>();
                (void)mean.item<float>();
                (void)max_val.item<float>();
            }

            // Test multiple steps (simulating RNN unrolling)
            if (offset < Size && Data[offset] % 4 == 0) {
                torch::Tensor h_next = output;
                for (int step = 0; step < 3; step++) {
                    if (use_bias) {
                        h_next = torch::rnn_relu_cell(input, h_next, w_ih, w_hh, b_ih, b_hh);
                    } else {
                        h_next = torch::rnn_relu_cell(input, h_next, w_ih, w_hh);
                    }
                }
                (void)h_next.sum().item<float>();
            }
        }
        catch (const c10::Error&) {
            // Silently catch shape mismatch or other tensor operation errors
        }
        catch (const std::runtime_error&) {
            // Silently catch runtime errors from invalid operations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}