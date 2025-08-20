#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have some bytes left for RNN parameters
        if (offset + 8 >= Size) {
            return 0;
        }
        
        // Extract RNN parameters from the input data
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        int64_t num_layers = 0;
        bool bias = false;
        bool batch_first = false;
        double dropout = 0.0;
        bool bidirectional = false;
        
        // Parse input_size (ensure it's positive)
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&input_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            input_size = std::abs(input_size) % 100 + 1; // Ensure positive and reasonable
        } else {
            input_size = 10; // Default
        }
        
        // Parse hidden_size (ensure it's positive)
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&hidden_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            hidden_size = std::abs(hidden_size) % 100 + 1; // Ensure positive and reasonable
        } else {
            hidden_size = 20; // Default
        }
        
        // Parse num_layers (ensure it's positive)
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&num_layers, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_layers = std::abs(num_layers) % 5 + 1; // Ensure positive and reasonable
        } else {
            num_layers = 1; // Default
        }
        
        // Parse boolean parameters
        if (offset < Size) {
            bias = Data[offset++] & 1;
        }
        
        if (offset < Size) {
            batch_first = Data[offset++] & 1;
        }
        
        // Parse dropout (ensure it's between 0 and 1)
        if (offset + sizeof(double) <= Size) {
            memcpy(&dropout, Data + offset, sizeof(double));
            offset += sizeof(double);
            dropout = std::abs(dropout);
            dropout = dropout - std::floor(dropout); // Get fractional part
        }
        
        if (offset < Size) {
            bidirectional = Data[offset++] & 1;
        }
        
        // Create RNN module
        torch::nn::RNNOptions options(input_size, hidden_size);
        options.num_layers(num_layers)
               .bias(bias)
               .batch_first(batch_first)
               .dropout(dropout)
               .bidirectional(bidirectional);
        
        torch::nn::RNN rnn_module(options);
        
        // Reshape input tensor if needed to match RNN expectations
        auto input_sizes = input_tensor.sizes().vec();
        
        // RNN expects input of shape (seq_len, batch, input_size) or (batch, seq_len, input_size)
        // We need to ensure the tensor has at least 3 dimensions
        if (input_tensor.dim() < 3) {
            // Add dimensions as needed
            while (input_tensor.dim() < 3) {
                input_tensor = input_tensor.unsqueeze(0);
            }
        }
        
        // Ensure the last dimension matches input_size
        if (input_tensor.size(-1) != input_size) {
            auto new_sizes = input_tensor.sizes().vec();
            new_sizes[input_tensor.dim() - 1] = input_size;
            input_tensor = input_tensor.reshape(new_sizes);
        }
        
        // Create initial hidden state
        torch::Tensor h0;
        int64_t num_directions = bidirectional ? 2 : 1;
        
        if (batch_first) {
            h0 = torch::zeros({num_layers * num_directions, input_tensor.size(0), hidden_size});
        } else {
            h0 = torch::zeros({num_layers * num_directions, input_tensor.size(1), hidden_size});
        }
        
        // Forward pass through RNN
        auto output = rnn_module->forward(input_tensor, h0);
        
        // Access output tensor and final hidden state
        auto output_tensor = std::get<0>(output);
        auto h_n = std::get<1>(output);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output_tensor.sum() + h_n.sum();
        
        // Prevent compiler from optimizing away the computation
        if (sum.item<float>() == -12345.6789f) {
            std::cerr << "Unlikely sum value encountered" << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}