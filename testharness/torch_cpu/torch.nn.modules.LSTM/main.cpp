#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least some bytes left for LSTM parameters
        if (offset + 8 >= Size) {
            return 0;
        }
        
        // Parse LSTM parameters from the input data
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        int64_t num_layers = 0;
        bool bias = false;
        bool batch_first = false;
        double dropout = 0.0;
        bool bidirectional = false;
        
        // Extract input_size (ensure it's positive)
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&input_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            input_size = std::abs(input_size) % 100 + 1; // Ensure positive and reasonable
        } else {
            input_size = 10; // Default
        }
        
        // Extract hidden_size (ensure it's positive)
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&hidden_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            hidden_size = std::abs(hidden_size) % 100 + 1; // Ensure positive and reasonable
        } else {
            hidden_size = 20; // Default
        }
        
        // Extract num_layers (ensure it's positive)
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&num_layers, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_layers = std::abs(num_layers) % 5 + 1; // Ensure positive and reasonable
        } else {
            num_layers = 1; // Default
        }
        
        // Extract boolean parameters
        if (offset < Size) {
            bias = Data[offset++] & 1;
        }
        
        if (offset < Size) {
            batch_first = Data[offset++] & 1;
        }
        
        // Extract dropout (ensure it's between 0 and 1)
        if (offset + sizeof(double) <= Size) {
            memcpy(&dropout, Data + offset, sizeof(double));
            offset += sizeof(double);
            dropout = std::abs(dropout);
            dropout = dropout - std::floor(dropout); // Get fractional part
        }
        
        if (offset < Size) {
            bidirectional = Data[offset++] & 1;
        }
        
        // Create LSTM module
        torch::nn::LSTM lstm_module(
            torch::nn::LSTMOptions(input_size, hidden_size)
                .num_layers(num_layers)
                .bias(bias)
                .batch_first(batch_first)
                .dropout(dropout)
                .bidirectional(bidirectional)
        );
        
        // Reshape input tensor if needed to match LSTM expectations
        // LSTM expects [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first=true
        auto input_sizes = input_tensor.sizes().vec();
        
        // If tensor is empty or scalar, create a valid tensor
        if (input_sizes.empty()) {
            // Create a minimal valid input: [1, 1, input_size]
            input_tensor = torch::ones({1, 1, input_size});
        } else if (input_sizes.size() == 1) {
            // 1D tensor: treat as single feature, add seq_len and batch dimensions
            input_tensor = input_tensor.reshape({1, 1, input_sizes[0]});
        } else if (input_sizes.size() == 2) {
            // 2D tensor: add batch dimension
            input_tensor = input_tensor.reshape({input_sizes[0], input_sizes[1], 1});
            // Ensure last dimension is input_size
            input_tensor = torch::nn::functional::pad(
                input_tensor, 
                torch::nn::functional::PadFuncOptions({0, input_size - 1, 0, 0, 0, 0})
            );
        } else if (input_sizes.size() >= 3) {
            // 3D+ tensor: reshape to ensure last dimension is input_size
            std::vector<int64_t> new_shape = {1, 1, input_size};
            for (size_t i = 0; i < std::min(size_t(2), input_sizes.size()); ++i) {
                new_shape[i] = input_sizes[i];
            }
            input_tensor = input_tensor.reshape(new_shape);
            
            // Ensure last dimension is input_size
            if (new_shape[2] != input_size) {
                input_tensor = torch::nn::functional::pad(
                    input_tensor, 
                    torch::nn::functional::PadFuncOptions({0, input_size - new_shape[2], 0, 0, 0, 0})
                );
            }
        }
        
        // Create initial hidden state (h0, c0)
        // h0 and c0 shape: [num_layers * num_directions, batch, hidden_size]
        int64_t num_directions = bidirectional ? 2 : 1;
        int64_t batch_size = batch_first ? input_tensor.size(0) : input_tensor.size(1);
        
        torch::Tensor h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
        torch::Tensor c0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
        
        // Forward pass through LSTM
        auto output = lstm_module->forward(input_tensor, std::make_tuple(h0, c0));
        
        // Unpack output
        auto output_tensor = std::get<0>(output);
        auto hidden_state = std::get<1>(output);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output_tensor.sum();
        auto h_sum = std::get<0>(hidden_state).sum();
        auto c_sum = std::get<1>(hidden_state).sum();
        
        // Combine all sums to prevent optimization from removing the computation
        auto total_sum = sum + h_sum + c_sum;
        
        // Access a value to ensure the tensor is materialized
        float value = total_sum.item<float>();
        (void)value; // Suppress unused variable warning
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
