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
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract dimensions from fuzz data
        int64_t batch_size = (Data[offset++] % 8) + 1;      // 1-8
        int64_t input_size = (Data[offset++] % 32) + 1;     // 1-32
        int64_t hidden_size = (Data[offset++] % 32) + 1;    // 1-32
        bool use_bias = Data[offset++] % 2 == 0;

        // Create input tensor: (batch, input_size)
        torch::Tensor input = torch::randn({batch_size, input_size});

        // Create hidden state tensor: (batch, hidden_size)
        torch::Tensor hx = torch::randn({batch_size, hidden_size});

        // Create weight tensors for GRU cell
        // GRU has 3 gates (reset, update, new), so weight matrices are 3*hidden_size
        torch::Tensor w_ih = torch::randn({3 * hidden_size, input_size});
        torch::Tensor w_hh = torch::randn({3 * hidden_size, hidden_size});

        // Optionally modify tensors with fuzz data if available
        if (offset + 4 <= Size) {
            float scale = static_cast<float>(Data[offset++]) / 255.0f * 2.0f;
            input = input * scale;
        }

        if (offset + 4 <= Size) {
            float scale = static_cast<float>(Data[offset++]) / 255.0f * 2.0f;
            hx = hx * scale;
        }

        torch::Tensor output;

        if (use_bias) {
            // Create bias tensors
            torch::Tensor b_ih = torch::randn({3 * hidden_size});
            torch::Tensor b_hh = torch::randn({3 * hidden_size});

            // Apply GRU cell with bias
            output = torch::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
        } else {
            // Apply GRU cell without bias (pass empty tensors)
            torch::Tensor b_ih = torch::Tensor();
            torch::Tensor b_hh = torch::Tensor();
            output = torch::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
        }

        // Verify output shape
        if (output.dim() != 2 || output.size(0) != batch_size || output.size(1) != hidden_size) {
            std::cerr << "Unexpected output shape" << std::endl;
        }

        // Use the output to prevent optimization
        volatile float sum = output.sum().item<float>();
        (void)sum;

        // Test with different input patterns based on fuzz data
        if (offset < Size) {
            uint8_t test_mode = Data[offset++] % 4;
            
            try {
                torch::Tensor test_input;
                torch::Tensor test_hx;

                switch (test_mode) {
                    case 0:
                        // Zero input
                        test_input = torch::zeros({batch_size, input_size});
                        test_hx = hx.clone();
                        break;
                    case 1:
                        // Zero hidden state
                        test_input = input.clone();
                        test_hx = torch::zeros({batch_size, hidden_size});
                        break;
                    case 2:
                        // Large values
                        test_input = torch::randn({batch_size, input_size}) * 100.0f;
                        test_hx = torch::randn({batch_size, hidden_size}) * 100.0f;
                        break;
                    case 3:
                        // Small values
                        test_input = torch::randn({batch_size, input_size}) * 0.001f;
                        test_hx = torch::randn({batch_size, hidden_size}) * 0.001f;
                        break;
                }

                torch::Tensor test_output;
                if (use_bias) {
                    torch::Tensor b_ih = torch::randn({3 * hidden_size});
                    torch::Tensor b_hh = torch::randn({3 * hidden_size});
                    test_output = torch::gru_cell(test_input, test_hx, w_ih, w_hh, b_ih, b_hh);
                } else {
                    test_output = torch::gru_cell(test_input, test_hx, w_ih, w_hh);
                }

                volatile float test_sum = test_output.sum().item<float>();
                (void)test_sum;
            }
            catch (...) {
                // Silently ignore expected failures in secondary tests
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