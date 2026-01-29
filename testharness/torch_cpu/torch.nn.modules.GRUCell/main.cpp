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
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract GRUCell parameters from fuzzer data first
        int64_t input_size = (Data[offset++] % 32) + 1;   // 1 to 32
        int64_t hidden_size = (Data[offset++] % 32) + 1;  // 1 to 32
        int64_t batch_size = (Data[offset++] % 8) + 1;    // 1 to 8
        bool bias = Data[offset++] & 1;
        
        // Create GRUCell with determined parameters
        torch::nn::GRUCell gru_cell(
            torch::nn::GRUCellOptions(input_size, hidden_size).bias(bias)
        );
        
        // Create input tensor with correct shape [batch_size, input_size]
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
            // Reshape to required dimensions
            int64_t total_elements = input.numel();
            if (total_elements == 0) {
                input = torch::randn({batch_size, input_size});
            } else if (total_elements < batch_size * input_size) {
                // Repeat to get enough elements
                int64_t repeat_factor = (batch_size * input_size + total_elements - 1) / total_elements;
                input = input.flatten().repeat({repeat_factor}).slice(0, 0, batch_size * input_size);
                input = input.reshape({batch_size, input_size});
            } else {
                input = input.flatten().slice(0, 0, batch_size * input_size).reshape({batch_size, input_size});
            }
        } else {
            input = torch::randn({batch_size, input_size});
        }
        
        // Ensure input is float type
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create hidden state tensor with correct shape [batch_size, hidden_size]
        torch::Tensor hx;
        if (offset < Size) {
            hx = fuzzer_utils::createTensor(Data, Size, offset);
            int64_t total_elements = hx.numel();
            if (total_elements == 0) {
                hx = torch::zeros({batch_size, hidden_size});
            } else if (total_elements < batch_size * hidden_size) {
                int64_t repeat_factor = (batch_size * hidden_size + total_elements - 1) / total_elements;
                hx = hx.flatten().repeat({repeat_factor}).slice(0, 0, batch_size * hidden_size);
                hx = hx.reshape({batch_size, hidden_size});
            } else {
                hx = hx.flatten().slice(0, 0, batch_size * hidden_size).reshape({batch_size, hidden_size});
            }
        } else {
            hx = torch::zeros({batch_size, hidden_size});
        }
        
        // Ensure hx is float type and matches input dtype
        if (hx.dtype() != input.dtype()) {
            hx = hx.to(input.dtype());
        }
        
        // Test forward pass with hidden state
        torch::Tensor output;
        try {
            output = gru_cell->forward(input, hx);
        } catch (...) {
            // Shape mismatch or similar - silently ignore
            return 0;
        }
        
        // Test forward pass without explicit hidden state (uses zeros)
        torch::Tensor output_no_hx;
        try {
            output_no_hx = gru_cell->forward(input);
        } catch (...) {
            // Silently ignore
        }
        
        // Verify output shape is [batch_size, hidden_size]
        if (output.dim() != 2 || output.size(0) != batch_size || output.size(1) != hidden_size) {
            std::cerr << "Unexpected output shape" << std::endl;
            return -1;
        }
        
        // Exercise output to ensure computation happened
        auto sum = output.sum();
        auto mean = output.mean();
        
        // Use volatile to prevent optimization
        volatile float s = sum.item<float>();
        volatile float m = mean.item<float>();
        (void)s;
        (void)m;
        
        // Test chaining - feed output as next hidden state
        try {
            torch::Tensor input2 = torch::randn({batch_size, input_size});
            torch::Tensor output2 = gru_cell->forward(input2, output);
            volatile float s2 = output2.sum().item<float>();
            (void)s2;
        } catch (...) {
            // Silently ignore chaining errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}