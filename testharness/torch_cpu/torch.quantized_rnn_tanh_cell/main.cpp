#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cmath>

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
        // Need sufficient data for parameters
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract dimensions from fuzzer data
        int64_t batch_size = 1 + (Data[offset++] % 8);      // 1-8
        int64_t input_size = 1 + (Data[offset++] % 32);     // 1-32
        int64_t hidden_size = 1 + (Data[offset++] % 32);    // 1-32
        
        // Create input tensor: [batch, input_size]
        torch::Tensor input = torch::randn({batch_size, input_size}, torch::kFloat32);
        
        // Create hidden state tensor: [batch, hidden_size]
        torch::Tensor hx = torch::randn({batch_size, hidden_size}, torch::kFloat32);
        
        // Create weight tensors with correct shapes
        // w_ih: [hidden_size, input_size]
        // w_hh: [hidden_size, hidden_size]
        torch::Tensor w_ih = torch::randn({hidden_size, input_size}, torch::kFloat32);
        torch::Tensor w_hh = torch::randn({hidden_size, hidden_size}, torch::kFloat32);
        
        // Create bias tensors: [hidden_size]
        torch::Tensor b_ih = torch::randn({hidden_size}, torch::kFloat32);
        torch::Tensor b_hh = torch::randn({hidden_size}, torch::kFloat32);
        
        // For quantized operations, we need quantized packed weights
        // These are typically created by quantization utilities
        // packed_ih and packed_hh should be the quantized/packed versions
        torch::Tensor packed_ih = torch::randn({hidden_size, input_size}, torch::kFloat32);
        torch::Tensor packed_hh = torch::randn({hidden_size, hidden_size}, torch::kFloat32);
        
        // Column offsets for quantized matmul
        torch::Tensor col_offsets_ih = torch::zeros({hidden_size}, torch::kInt32);
        torch::Tensor col_offsets_hh = torch::zeros({hidden_size}, torch::kInt32);
        
        // Parse scale and zero point parameters
        double scale_ih = 0.1;
        double scale_hh = 0.1;
        int64_t zero_point_ih = 0;
        int64_t zero_point_hh = 0;
        
        if (offset + sizeof(float) <= Size) {
            float tmp;
            std::memcpy(&tmp, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(tmp) && tmp != 0.0f) {
                scale_ih = std::abs(tmp);
            }
        }
        
        if (offset + sizeof(float) <= Size) {
            float tmp;
            std::memcpy(&tmp, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(tmp) && tmp != 0.0f) {
                scale_hh = std::abs(tmp);
            }
        }
        
        if (offset + 1 <= Size) {
            zero_point_ih = static_cast<int64_t>(Data[offset++]) - 128;
        }
        
        if (offset + 1 <= Size) {
            zero_point_hh = static_cast<int64_t>(Data[offset++]) - 128;
        }
        
        // Seed random with fuzzer data for reproducibility
        if (offset + 4 <= Size) {
            uint32_t seed;
            std::memcpy(&seed, Data + offset, sizeof(uint32_t));
            torch::manual_seed(seed);
            offset += sizeof(uint32_t);
        }
        
        // Call the quantized_rnn_tanh_cell function
        // Note: This is a low-level quantized operation
        try {
            torch::Tensor result = torch::quantized_rnn_tanh_cell(
                input, hx, w_ih, w_hh, b_ih, b_hh,
                packed_ih, packed_hh, col_offsets_ih, col_offsets_hh,
                scale_ih, scale_hh, zero_point_ih, zero_point_hh
            );
            
            // Verify result
            if (result.defined()) {
                volatile float val = result.sum().item<float>();
                (void)val;
            }
        }
        catch (const c10::Error&) {
            // Expected failures due to quantization constraints - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}