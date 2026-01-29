#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <algorithm>

// --- Fuzzer Entry Point ---
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
        // Need minimum data for tensor dimensions and parameters
        if (Size < 16) return 0;

        size_t offset = 0;

        // Extract dimensions from fuzzer data for consistent tensor shapes
        // LSTM cell needs compatible dimensions:
        // - input: (batch, input_size)
        // - h, c: (batch, hidden_size)
        // - w_ih: (4*hidden_size, input_size)
        // - w_hh: (4*hidden_size, hidden_size)
        // - biases: (4*hidden_size,)
        int batch_size = std::max(1, static_cast<int>(Data[offset++] % 8) + 1);
        int input_size = std::max(1, static_cast<int>(Data[offset++] % 16) + 1);
        int hidden_size = std::max(1, static_cast<int>(Data[offset++] % 16) + 1);
        int gate_size = 4 * hidden_size;

        // Create input tensor: (batch_size, input_size)
        torch::Tensor input = torch::randn({batch_size, input_size});

        // Create hidden state tensors: h and c both (batch_size, hidden_size)
        torch::Tensor h_state = torch::randn({batch_size, hidden_size});
        torch::Tensor c_state = torch::randn({batch_size, hidden_size});

        // Weight tensors:
        // w_ih: (4*hidden_size, input_size), w_hh: (4*hidden_size, hidden_size)
        torch::Tensor w_ih = torch::randn({gate_size, input_size});
        torch::Tensor w_hh = torch::randn({gate_size, hidden_size});

        // Bias tensors: (4*hidden_size,)
        torch::Tensor b_ih = torch::randn({gate_size});
        torch::Tensor b_hh = torch::randn({gate_size});

        // Packed weight tensors (quantized int8)
        torch::Tensor packed_ih = torch::randint(0, 256, {gate_size, input_size}, torch::kInt8);
        torch::Tensor packed_hh = torch::randint(0, 256, {gate_size, hidden_size}, torch::kInt8);

        // Column offsets: (4*hidden_size,)
        torch::Tensor col_offsets_ih = torch::randint(-128, 128, {gate_size}, torch::kInt32);
        torch::Tensor col_offsets_hh = torch::randint(-128, 128, {gate_size}, torch::kInt32);

        // Extract scale and zero point values from fuzzer data
        // Use reasonable ranges to avoid numerical issues
        float scale_ih_val = 0.001f + static_cast<float>(Data[offset++ % Size] % 100) * 0.001f;
        float scale_hh_val = 0.001f + static_cast<float>(Data[offset++ % Size] % 100) * 0.001f;
        int64_t zp_ih_val = static_cast<int64_t>(Data[offset++ % Size] % 256) - 128;
        int64_t zp_hh_val = static_cast<int64_t>(Data[offset++ % Size] % 256) - 128;

        // Create Scalar values for API
        at::Scalar scale_ih(scale_ih_val);
        at::Scalar scale_hh(scale_hh_val);
        at::Scalar zero_point_ih(zp_ih_val);
        at::Scalar zero_point_hh(zp_hh_val);

        // Hidden state as TensorList
        std::vector<torch::Tensor> hx = {h_state, c_state};

        // Try calling quantized_lstm_cell
        // Note: May fail if FBGEMM not supported or packed weights invalid
        try {
            auto result = torch::quantized_lstm_cell(
                input,
                hx,
                w_ih, w_hh, b_ih, b_hh,
                packed_ih, packed_hh,
                col_offsets_ih, col_offsets_hh,
                scale_ih, scale_hh, zero_point_ih, zero_point_hh
            );

            auto [hy, cy] = result;
            (void)hy.numel();
            (void)cy.numel();
        } catch (...) {
            // Expected failures with invalid quantized weights - continue silently
        }

        // Try with zero biases
        try {
            auto zero_b_ih = torch::zeros({gate_size});
            auto zero_b_hh = torch::zeros({gate_size});

            auto result2 = torch::quantized_lstm_cell(
                input,
                hx,
                w_ih, w_hh, zero_b_ih, zero_b_hh,
                packed_ih, packed_hh,
                col_offsets_ih, col_offsets_hh,
                scale_ih, scale_hh, zero_point_ih, zero_point_hh
            );
            (void)std::get<0>(result2).numel();
        } catch (...) {
            // Silently continue
        }

        // Try with different scale values
        try {
            at::Scalar small_scale(0.0001);
            at::Scalar one_scale(1.0);
            at::Scalar zero_zp(static_cast<int64_t>(0));

            auto result3 = torch::quantized_lstm_cell(
                input,
                hx,
                w_ih, w_hh, b_ih, b_hh,
                packed_ih, packed_hh,
                col_offsets_ih, col_offsets_hh,
                small_scale, one_scale, zero_zp, zero_zp
            );
            (void)std::get<0>(result3).numel();
        } catch (...) {
            // Silently continue
        }

        // Try with empty bias tensors
        try {
            auto result4 = torch::quantized_lstm_cell(
                input,
                hx,
                w_ih, w_hh,
                torch::Tensor(), torch::Tensor(),
                packed_ih, packed_hh,
                col_offsets_ih, col_offsets_hh,
                scale_ih, scale_hh, zero_point_ih, zero_point_hh
            );
            (void)std::get<0>(result4).numel();
        } catch (...) {
            // Silently continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}