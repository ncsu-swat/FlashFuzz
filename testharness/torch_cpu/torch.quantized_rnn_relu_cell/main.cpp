#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>
#include <cstring>

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
        size_t offset = 0;
        
        // Need sufficient data for parameter generation
        if (Size < 32) {
            return 0;
        }
        
        // Parse dimensions from fuzzer data
        int batch_size = 1 + (Data[offset++] % 8);     // 1-8
        int input_size = 1 + (Data[offset++] % 32);    // 1-32
        int hidden_size = 1 + (Data[offset++] % 32);   // 1-32
        
        // Create input tensor: [batch_size, input_size]
        torch::Tensor input = torch::randn({batch_size, input_size}, torch::kFloat32);
        
        // Create hidden state tensor: [batch_size, hidden_size]
        torch::Tensor hx = torch::randn({batch_size, hidden_size}, torch::kFloat32);
        
        // Create weight tensors with proper dimensions
        // w_ih: [hidden_size, input_size]
        torch::Tensor w_ih = torch::randn({hidden_size, input_size}, torch::kFloat32);
        // w_hh: [hidden_size, hidden_size]
        torch::Tensor w_hh = torch::randn({hidden_size, hidden_size}, torch::kFloat32);
        
        // Create bias tensors: [hidden_size]
        torch::Tensor b_ih = torch::randn({hidden_size}, torch::kFloat32);
        torch::Tensor b_hh = torch::randn({hidden_size}, torch::kFloat32);
        
        // Create packed weight tensors (quantized int8)
        // These should be properly quantized packed weights
        torch::Tensor packed_ih = torch::randint(-128, 127, {hidden_size, input_size}, torch::kInt8);
        torch::Tensor packed_hh = torch::randint(-128, 127, {hidden_size, hidden_size}, torch::kInt8);
        
        // Create column offset tensors: [hidden_size]
        torch::Tensor col_offsets_ih = torch::randint(-128, 127, {hidden_size}, torch::kInt32);
        torch::Tensor col_offsets_hh = torch::randint(-128, 127, {hidden_size}, torch::kInt32);
        
        // Parse scale parameters from fuzzer data
        double scale_ih = 1.0;
        double scale_hh = 1.0;
        int64_t zero_point_ih = 0;
        int64_t zero_point_hh = 0;
        
        if (offset + sizeof(float) * 2 <= Size) {
            float f_scale_ih, f_scale_hh;
            std::memcpy(&f_scale_ih, Data + offset, sizeof(float));
            offset += sizeof(float);
            std::memcpy(&f_scale_hh, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Sanitize scales
            if (std::isfinite(f_scale_ih)) {
                scale_ih = std::abs(f_scale_ih);
            }
            if (std::isfinite(f_scale_hh)) {
                scale_hh = std::abs(f_scale_hh);
            }
        }
        
        // Clamp scales to reasonable range
        scale_ih = std::max(1e-6, std::min(1e6, scale_ih));
        scale_hh = std::max(1e-6, std::min(1e6, scale_hh));
        
        // Parse zero points
        if (offset + 2 <= Size) {
            zero_point_ih = static_cast<int64_t>(Data[offset++]) - 128;
            zero_point_hh = static_cast<int64_t>(Data[offset++]) - 128;
        }
        
        // Randomize tensor values based on fuzzer data to increase coverage
        if (offset < Size) {
            float scale_factor = (Data[offset++] % 100) / 10.0f + 0.1f;
            input = input * scale_factor;
            hx = hx * scale_factor;
        }
        
        // Inner try-catch for expected failures (shape mismatches, etc.)
        try {
            // Call quantized_rnn_relu_cell
            auto result = torch::quantized_rnn_relu_cell(
                input, hx, w_ih, w_hh, b_ih, b_hh,
                packed_ih, packed_hh, col_offsets_ih, col_offsets_hh,
                scale_ih, scale_hh, zero_point_ih, zero_point_hh);
            
            // Access result to ensure computation completes
            (void)result.numel();
        }
        catch (const c10::Error &e) {
            // Expected errors from invalid combinations - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}