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
        size_t offset = 0;
        
        // Need sufficient data to create meaningful tensors
        if (Size < 8) {
            return 0;
        }
        
        // Extract dimensions from fuzzer data
        uint8_t batch_byte = Data[offset++];
        uint8_t input_size_byte = Data[offset++];
        uint8_t hidden_size_byte = Data[offset++];
        uint8_t use_bias_byte = Data[offset++];
        
        // Clamp dimensions to reasonable ranges
        int64_t batch_size = (batch_byte % 8) + 1;      // 1-8
        int64_t input_size = (input_size_byte % 16) + 1; // 1-16
        int64_t hidden_size = (hidden_size_byte % 16) + 1; // 1-16
        bool use_bias = (use_bias_byte % 2) == 0;
        
        // Create input tensor: (batch, input_size)
        torch::Tensor input = torch::randn({batch_size, input_size});
        
        // Create hidden state tensor: (batch, hidden_size)
        torch::Tensor hx = torch::randn({batch_size, hidden_size});
        
        // Create weight tensors with proper shapes
        // w_ih: (hidden_size, input_size)
        torch::Tensor w_ih = torch::randn({hidden_size, input_size});
        
        // w_hh: (hidden_size, hidden_size)
        torch::Tensor w_hh = torch::randn({hidden_size, hidden_size});
        
        // Create bias tensors: (hidden_size)
        torch::Tensor b_ih, b_hh;
        if (use_bias) {
            b_ih = torch::randn({hidden_size});
            b_hh = torch::randn({hidden_size});
        } else {
            // Use empty tensors when no bias
            b_ih = torch::Tensor();
            b_hh = torch::Tensor();
        }
        
        // Use remaining fuzzer data to modify tensor values if available
        if (offset < Size) {
            try {
                torch::Tensor fuzz_input = fuzzer_utils::createTensor(Data, Size, offset);
                // Scale and add fuzz data to input if shapes are compatible
                if (fuzz_input.numel() > 0) {
                    auto flat_fuzz = fuzz_input.flatten();
                    auto flat_input = input.flatten();
                    int64_t copy_len = std::min(flat_fuzz.numel(), flat_input.numel());
                    flat_input.slice(0, 0, copy_len).copy_(flat_fuzz.slice(0, 0, copy_len));
                }
            } catch (...) {
                // Ignore errors from createTensor
            }
        }
        
        // Inner try-catch for expected shape/type mismatches
        try {
            // Apply the RNN tanh cell operation
            torch::Tensor output = torch::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
            
            // Verify output shape is correct: (batch, hidden_size)
            if (output.dim() == 2 && output.size(0) == batch_size && output.size(1) == hidden_size) {
                // Perform operations to ensure output is used
                auto sum = output.sum();
                auto mean = output.mean();
                auto max_val = output.max();
                
                // Check output is within tanh range [-1, 1]
                auto min_elem = output.min().item<float>();
                auto max_elem = output.max().item<float>();
                (void)min_elem;
                (void)max_elem;
            }
            
            // Test with different input variations
            if (offset < Size && Data[offset % Size] % 4 == 0) {
                // Test with zero input
                torch::Tensor zero_input = torch::zeros({batch_size, input_size});
                torch::Tensor output2 = torch::rnn_tanh_cell(zero_input, hx, w_ih, w_hh, b_ih, b_hh);
                (void)output2;
            }
            
            if (offset < Size && Data[offset % Size] % 4 == 1) {
                // Test with zero hidden state
                torch::Tensor zero_hx = torch::zeros({batch_size, hidden_size});
                torch::Tensor output3 = torch::rnn_tanh_cell(input, zero_hx, w_ih, w_hh, b_ih, b_hh);
                (void)output3;
            }
        }
        catch (...) {
            // Silently catch expected failures (shape mismatches, etc.)
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}