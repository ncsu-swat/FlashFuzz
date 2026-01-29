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
        // Need enough data to derive dimensions
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Derive dimensions from fuzzer data
        int64_t batch_size = 1 + (Data[offset++] % 16);      // 1-16
        int64_t input_size = 1 + (Data[offset++] % 32);      // 1-32
        int64_t hidden_size = 1 + (Data[offset++] % 32);     // 1-32
        bool use_bias = Data[offset++] % 2 == 0;
        
        // Create input tensor: (batch, input_size)
        torch::Tensor input = torch::randn({batch_size, input_size});
        
        // Create hidden state tensor: (batch, hidden_size)
        torch::Tensor hx = torch::randn({batch_size, hidden_size});
        
        // Create weight tensors with correct shapes for GRU
        // GRU has 3 gates (reset, update, new), so weights are 3*hidden_size
        torch::Tensor w_ih = torch::randn({3 * hidden_size, input_size});
        torch::Tensor w_hh = torch::randn({3 * hidden_size, hidden_size});
        
        // Optionally use fuzzer data to perturb weights
        if (offset + 4 <= Size) {
            float scale = 0.1f + (Data[offset++] % 100) / 50.0f;  // 0.1 to 2.1
            w_ih = w_ih * scale;
        }
        if (offset + 4 <= Size) {
            float scale = 0.1f + (Data[offset++] % 100) / 50.0f;
            w_hh = w_hh * scale;
        }
        
        // Perturb input and hidden state with fuzzer data
        if (offset < Size) {
            torch::Tensor noise_input = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                // Try to add noise if shapes are compatible
                if (noise_input.numel() > 0) {
                    noise_input = noise_input.to(torch::kFloat).flatten();
                    int64_t use_elems = std::min(noise_input.numel(), input.numel());
                    input.flatten().slice(0, 0, use_elems).add_(
                        noise_input.slice(0, 0, use_elems) * 0.1f);
                }
            } catch (...) {
                // Silently ignore shape/type mismatches
            }
        }
        
        if (offset < Size) {
            torch::Tensor noise_hx = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                if (noise_hx.numel() > 0) {
                    noise_hx = noise_hx.to(torch::kFloat).flatten();
                    int64_t use_elems = std::min(noise_hx.numel(), hx.numel());
                    hx.flatten().slice(0, 0, use_elems).add_(
                        noise_hx.slice(0, 0, use_elems) * 0.1f);
                }
            } catch (...) {
                // Silently ignore shape/type mismatches
            }
        }
        
        torch::Tensor output;
        
        if (use_bias) {
            // Create bias tensors: (3 * hidden_size)
            torch::Tensor b_ih = torch::randn({3 * hidden_size});
            torch::Tensor b_hh = torch::randn({3 * hidden_size});
            
            // Perturb biases with fuzzer data
            if (offset < Size) {
                float bias_scale = (Data[offset++] % 100) / 50.0f;
                b_ih = b_ih * bias_scale;
            }
            if (offset < Size) {
                float bias_scale = (Data[offset++] % 100) / 50.0f;
                b_hh = b_hh * bias_scale;
            }
            
            output = torch::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
        } else {
            output = torch::gru_cell(input, hx, w_ih, w_hh);
        }
        
        // Verify output shape is correct: (batch, hidden_size)
        if (output.dim() != 2 || output.size(0) != batch_size || output.size(1) != hidden_size) {
            std::cerr << "Unexpected output shape" << std::endl;
            return -1;
        }
        
        // Exercise the output
        auto sum = output.sum();
        auto mean = output.mean();
        
        // Additional operations to increase coverage
        auto detached = output.detach();
        auto cloned = output.clone();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}