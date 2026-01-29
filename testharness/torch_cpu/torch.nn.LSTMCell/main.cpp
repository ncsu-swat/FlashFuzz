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
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters from fuzzer data
        int64_t batch_size = (Data[offset++] % 8) + 1;    // 1-8
        int64_t input_size = (Data[offset++] % 16) + 1;   // 1-16
        int64_t hidden_size = (Data[offset++] % 16) + 1;  // 1-16
        bool use_bias = (Data[offset++] % 2 == 0);
        
        // Create input tensor with proper shape [batch, input_size]
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is the right shape for LSTMCell
        // LSTMCell expects input of shape [batch, input_size]
        int64_t total_elements = input.numel();
        if (total_elements == 0) {
            total_elements = batch_size * input_size;
            input = torch::randn({batch_size, input_size});
        } else {
            // Reshape to 2D [batch_size, input_size]
            input = input.flatten();
            int64_t needed = batch_size * input_size;
            if (input.numel() < needed) {
                // Pad with zeros
                input = torch::cat({input, torch::zeros(needed - input.numel(), input.options())});
            }
            input = input.slice(0, 0, needed).reshape({batch_size, input_size});
        }
        
        // Ensure float type for LSTM
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create hidden state tensors (h0, c0) with shape [batch, hidden_size]
        torch::Tensor h0 = torch::zeros({batch_size, hidden_size}, torch::kFloat32);
        torch::Tensor c0 = torch::zeros({batch_size, hidden_size}, torch::kFloat32);
        
        // If we have more data, use it to initialize h0 and c0
        if (offset < Size) {
            torch::Tensor h_init = fuzzer_utils::createTensor(Data, Size, offset);
            if (h_init.numel() > 0) {
                h_init = h_init.flatten().to(torch::kFloat32);
                int64_t h_needed = batch_size * hidden_size;
                if (h_init.numel() >= h_needed) {
                    h0 = h_init.slice(0, 0, h_needed).reshape({batch_size, hidden_size});
                }
            }
        }
        
        if (offset < Size) {
            torch::Tensor c_init = fuzzer_utils::createTensor(Data, Size, offset);
            if (c_init.numel() > 0) {
                c_init = c_init.flatten().to(torch::kFloat32);
                int64_t c_needed = batch_size * hidden_size;
                if (c_init.numel() >= c_needed) {
                    c0 = c_init.slice(0, 0, c_needed).reshape({batch_size, hidden_size});
                }
            }
        }
        
        // Create LSTM cell with input_size and hidden_size
        torch::nn::LSTMCellOptions options(input_size, hidden_size);
        options.bias(use_bias);
        
        torch::nn::LSTMCell lstm_cell(options);
        
        // Apply the LSTM cell
        auto result = lstm_cell->forward(input, std::make_tuple(h0, c0));
        
        // Extract the output tensors
        torch::Tensor h1 = std::get<0>(result);
        torch::Tensor c1 = std::get<1>(result);
        
        // Perform some operations on the output to ensure it's used
        auto sum_h = torch::sum(h1);
        auto sum_c = torch::sum(c1);
        
        // Prevent the compiler from optimizing away the computation
        if (sum_h.item<float>() == -12345.6789f && sum_c.item<float>() == -12345.6789f) {
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}