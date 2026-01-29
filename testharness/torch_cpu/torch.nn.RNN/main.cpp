#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes for basic parameters
        if (Size < 16) {
            return 0;
        }
        
        // Extract parameters for RNN first
        uint8_t input_size_byte = Data[offset++];
        int64_t input_size = (input_size_byte % 16) + 1; // 1-16 input size
        
        uint8_t hidden_size_byte = Data[offset++];
        int64_t hidden_size = (hidden_size_byte % 32) + 1; // 1-32 hidden size
        
        uint8_t num_layers_byte = Data[offset++];
        int64_t num_layers = (num_layers_byte % 3) + 1; // 1-3 layers
        
        uint8_t bias_byte = Data[offset++];
        bool bias = bias_byte % 2 == 1;
        
        uint8_t batch_first_byte = Data[offset++];
        bool batch_first = batch_first_byte % 2 == 1;
        
        uint8_t dropout_byte = Data[offset++];
        // Only use dropout if num_layers > 1 (required by PyTorch)
        double dropout = (num_layers > 1) ? (static_cast<double>(dropout_byte % 50) / 100.0) : 0.0;
        
        uint8_t bidirectional_byte = Data[offset++];
        bool bidirectional = bidirectional_byte % 2 == 1;
        
        uint8_t nonlinearity_byte = Data[offset++];
        // Use torch::kTanh and torch::kReLU for RNN nonlinearity
        auto nonlinearity = (nonlinearity_byte % 2 == 0) ? 
            torch::kTanh : torch::kReLU;
        
        // Extract sequence length and batch size from fuzz data
        uint8_t seq_len_byte = Data[offset++];
        int64_t seq_len = (seq_len_byte % 16) + 1; // 1-16 sequence length
        
        uint8_t batch_size_byte = Data[offset++];
        int64_t batch_size = (batch_size_byte % 8) + 1; // 1-8 batch size
        
        // Create RNN module with explicit input_size
        torch::nn::RNNOptions options(input_size, hidden_size);
        options.num_layers(num_layers)
               .bias(bias)
               .batch_first(batch_first)
               .dropout(dropout)
               .bidirectional(bidirectional)
               .nonlinearity(nonlinearity);
        
        auto rnn = torch::nn::RNN(options);
        
        // Create input tensor with correct shape
        // RNN expects (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first
        torch::Tensor input;
        if (batch_first) {
            input = torch::randn({batch_size, seq_len, input_size});
        } else {
            input = torch::randn({seq_len, batch_size, input_size});
        }
        
        // Use remaining fuzz data to perturb the input
        size_t remaining = Size - offset;
        if (remaining > 0) {
            auto input_flat = input.flatten();
            size_t num_elements = std::min(remaining, static_cast<size_t>(input_flat.numel()));
            for (size_t i = 0; i < num_elements; i++) {
                float val = static_cast<float>(Data[offset + i]) / 128.0f - 1.0f;
                input_flat[i] = val;
            }
            input = input_flat.view(input.sizes());
        }
        
        // Create initial hidden state
        int64_t num_directions = bidirectional ? 2 : 1;
        int64_t actual_batch_size = batch_first ? input.size(0) : input.size(1);
        torch::Tensor h0 = torch::zeros({num_layers * num_directions, actual_batch_size, hidden_size});
        
        // Test forward with hidden state
        try {
            auto output = rnn->forward(input, h0);
            auto output_tensor = std::get<0>(output);
            auto hidden_tensor = std::get<1>(output);
            
            // Verify output shapes
            auto sum = output_tensor.sum() + hidden_tensor.sum();
            (void)sum;
        } catch (const c10::Error&) {
            // Shape mismatch or similar, silently ignore
        }
        
        // Test forward without initial hidden state
        try {
            auto output2 = rnn->forward(input);
            auto output_tensor2 = std::get<0>(output2);
            auto sum2 = output_tensor2.sum();
            (void)sum2;
        } catch (const c10::Error&) {
            // Silently ignore expected errors
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}