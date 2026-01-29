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
        // Need sufficient data for dimensions and tensors
        if (Size < 20) {
            return 0;
        }

        size_t offset = 0;

        // Parse dimensions from fuzzer data
        int batch_size = 1 + (Data[offset++] % 8);      // 1-8
        int input_size = 1 + (Data[offset++] % 32);     // 1-32
        int hidden_size = 1 + (Data[offset++] % 32);    // 1-32

        // Create input tensor: (batch, input_size)
        torch::Tensor input = torch::randn({batch_size, input_size});

        // Create hidden state tensor: (batch, hidden_size)
        torch::Tensor hx = torch::randn({batch_size, hidden_size});

        // For quantized GRU cell, we need:
        // weight_ih: (3 * hidden_size, input_size)
        // weight_hh: (3 * hidden_size, hidden_size)
        int gate_size = 3 * hidden_size;

        // Create float weight tensors first, then quantize
        torch::Tensor weight_ih_float = torch::randn({gate_size, input_size});
        torch::Tensor weight_hh_float = torch::randn({gate_size, hidden_size});

        // Parse scale and zero point values
        double w_ih_scale = 0.01 + (Data[offset++ % Size] / 255.0) * 0.1;
        double w_hh_scale = 0.01 + (Data[offset++ % Size] / 255.0) * 0.1;
        int64_t w_ih_zero_point = Data[offset++ % Size] % 256;
        int64_t w_hh_zero_point = Data[offset++ % Size] % 256;

        // Quantize the weight tensors
        torch::Tensor weight_ih = torch::quantize_per_tensor(
            weight_ih_float, w_ih_scale, w_ih_zero_point, torch::kQInt8);
        torch::Tensor weight_hh = torch::quantize_per_tensor(
            weight_hh_float, w_hh_scale, w_hh_zero_point, torch::kQInt8);

        // Create bias tensors: (3 * hidden_size)
        torch::Tensor bias_ih = torch::randn({gate_size});
        torch::Tensor bias_hh = torch::randn({gate_size});

        // Create packed weight tensors (for FBGEMM, these are precomputed)
        // For testing, use the same quantized weights
        torch::Tensor packed_ih = weight_ih;
        torch::Tensor packed_hh = weight_hh;

        // Create column offsets tensors: (3 * hidden_size)
        torch::Tensor col_offsets_ih = torch::zeros({gate_size}, torch::kInt32);
        torch::Tensor col_offsets_hh = torch::zeros({gate_size}, torch::kInt32);

        // Call quantized_gru_cell
        try {
            torch::Tensor result = torch::quantized_gru_cell(
                input,
                hx,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                packed_ih,
                packed_hh,
                col_offsets_ih,
                col_offsets_hh,
                w_ih_scale,
                w_hh_scale,
                w_ih_zero_point,
                w_hh_zero_point
            );

            // Use the result to prevent optimization
            if (result.defined()) {
                volatile float sum_val = result.sum().item<float>();
                (void)sum_val;
            }
        }
        catch (const c10::Error &e) {
            // Silently catch expected errors (shape mismatches, unsupported configs)
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}