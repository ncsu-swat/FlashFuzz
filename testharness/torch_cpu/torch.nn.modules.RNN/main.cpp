#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least some bytes for parameters
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract RNN parameters from the input data
        int64_t input_size = (Data[offset++] % 32) + 1;    // 1-32
        int64_t hidden_size = (Data[offset++] % 32) + 1;   // 1-32
        int64_t num_layers = (Data[offset++] % 3) + 1;     // 1-3
        int64_t seq_len = (Data[offset++] % 16) + 1;       // 1-16
        int64_t batch_size = (Data[offset++] % 8) + 1;     // 1-8
        
        bool bias = Data[offset++] & 1;
        bool batch_first = Data[offset++] & 1;
        bool bidirectional = Data[offset++] & 1;
        
        // Dropout only effective when num_layers > 1
        double dropout = 0.0;
        if (num_layers > 1) {
            dropout = static_cast<double>(Data[offset++] % 50) / 100.0; // 0.0-0.49
        } else {
            offset++;
        }
        
        // Determine nonlinearity (tanh or relu) - use string representation
        bool use_relu = Data[offset++] & 1;
        
        // Create RNN module with options
        torch::nn::RNNOptions options(input_size, hidden_size);
        options.num_layers(num_layers)
               .bias(bias)
               .batch_first(batch_first)
               .dropout(dropout)
               .bidirectional(bidirectional);
        
        // Set nonlinearity using the appropriate enum
        if (use_relu) {
            options.nonlinearity(torch::kReLU);
        } else {
            options.nonlinearity(torch::kTanh);
        }
        
        torch::nn::RNN rnn_module(options);
        rnn_module->eval(); // Set to eval mode to disable dropout randomness
        
        // Create input tensor with correct shape
        // RNN expects (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first
        torch::Tensor input_tensor;
        if (batch_first) {
            input_tensor = torch::randn({batch_size, seq_len, input_size});
        } else {
            input_tensor = torch::randn({seq_len, batch_size, input_size});
        }
        
        // Use remaining fuzzer data to perturb the tensor values
        if (offset < Size) {
            size_t remaining = Size - offset;
            auto flat = input_tensor.flatten();
            int64_t num_elements = flat.numel();
            for (size_t i = 0; i < remaining && i < static_cast<size_t>(num_elements); i++) {
                flat[i] = static_cast<float>(Data[offset + i]) / 255.0f * 2.0f - 1.0f;
            }
            input_tensor = flat.view(input_tensor.sizes());
        }
        
        int64_t num_directions = bidirectional ? 2 : 1;
        
        // Test forward pass without initial hidden state
        try {
            auto output1 = rnn_module->forward(input_tensor);
            auto output_tensor1 = std::get<0>(output1);
            auto h_n1 = std::get<1>(output1);
            
            // Verify output shapes
            (void)output_tensor1.sum();
            (void)h_n1.sum();
        } catch (...) {
            // Expected failures for certain configurations
        }
        
        // Test forward pass with initial hidden state
        try {
            torch::Tensor h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
            
            auto output2 = rnn_module->forward(input_tensor, h0);
            auto output_tensor2 = std::get<0>(output2);
            auto h_n2 = std::get<1>(output2);
            
            // Verify outputs are computed
            auto sum = output_tensor2.sum() + h_n2.sum();
            
            // Prevent optimization
            if (sum.item<float>() == -12345.6789f) {
                std::cerr << "Unlikely value" << std::endl;
            }
        } catch (...) {
            // Expected failures for certain configurations
        }
        
        // Test with random initial hidden state
        try {
            torch::Tensor h0_rand = torch::randn({num_layers * num_directions, batch_size, hidden_size});
            
            auto output3 = rnn_module->forward(input_tensor, h0_rand);
            (void)std::get<0>(output3).sum();
        } catch (...) {
            // Expected failures
        }
        
        // Test flattening parameters (used for cuDNN optimization)
        try {
            rnn_module->flatten_parameters();
        } catch (...) {
            // May fail in some configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}