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
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract dimensions from fuzzer data
        int64_t batch_size = 1 + (Data[offset++] % 8);      // 1-8
        int64_t input_size = 1 + (Data[offset++] % 32);     // 1-32
        int64_t hidden_size = 1 + (Data[offset++] % 32);    // 1-32
        bool use_bias = Data[offset++] & 0x1;

        // Create input tensor: (batch, input_size)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        if (input.numel() == 0) {
            input = torch::randn({batch_size, input_size});
        } else {
            input = input.reshape({batch_size, input_size});
        }

        // Create hidden state tensors: (batch, hidden_size)
        torch::Tensor h0 = torch::randn({batch_size, hidden_size});
        torch::Tensor c0 = torch::randn({batch_size, hidden_size});

        // Create weight tensors with correct shapes
        // w_ih: (4*hidden_size, input_size)
        // w_hh: (4*hidden_size, hidden_size)
        torch::Tensor w_ih = torch::randn({4 * hidden_size, input_size});
        torch::Tensor w_hh = torch::randn({4 * hidden_size, hidden_size});

        // Create bias tensors: (4*hidden_size)
        torch::Tensor b_ih, b_hh;
        if (use_bias) {
            b_ih = torch::randn({4 * hidden_size});
            b_hh = torch::randn({4 * hidden_size});
        }

        // Apply some fuzzer-derived perturbation to weights
        if (offset < Size) {
            float scale = static_cast<float>(Data[offset++]) / 255.0f * 2.0f;
            w_ih = w_ih * scale;
            w_hh = w_hh * scale;
        }

        // Test lstm_cell with bias
        try {
            if (use_bias) {
                auto result = torch::lstm_cell(input, {h0, c0}, w_ih, w_hh, b_ih, b_hh);
                auto h_out = std::get<0>(result);
                auto c_out = std::get<1>(result);
                
                // Verify output shapes
                (void)h_out.sizes();
                (void)c_out.sizes();
            }
        } catch (const c10::Error& e) {
            // Expected failures, silently catch
        } catch (const std::exception& e) {
            // Other expected failures
        }

        // Test lstm_cell without bias
        try {
            auto result = torch::lstm_cell(input, {h0, c0}, w_ih, w_hh);
            auto h_out = std::get<0>(result);
            auto c_out = std::get<1>(result);
            
            // Verify output shapes
            (void)h_out.sizes();
            (void)c_out.sizes();
        } catch (const c10::Error& e) {
            // Expected failures, silently catch
        } catch (const std::exception& e) {
            // Other expected failures
        }

        // Test with different dtypes
        try {
            torch::Tensor input_f64 = input.to(torch::kFloat64);
            torch::Tensor h0_f64 = h0.to(torch::kFloat64);
            torch::Tensor c0_f64 = c0.to(torch::kFloat64);
            torch::Tensor w_ih_f64 = w_ih.to(torch::kFloat64);
            torch::Tensor w_hh_f64 = w_hh.to(torch::kFloat64);
            
            auto result = torch::lstm_cell(input_f64, {h0_f64, c0_f64}, w_ih_f64, w_hh_f64);
            (void)std::get<0>(result);
            (void)std::get<1>(result);
        } catch (const c10::Error& e) {
            // Expected failures
        } catch (const std::exception& e) {
            // Other expected failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}