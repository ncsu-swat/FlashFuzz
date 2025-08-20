#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Parse input parameters for LSTM
        int64_t input_size = (Data[offset] % 100) + 1;
        offset++;
        
        int64_t hidden_size = (Data[offset] % 100) + 1;
        offset++;
        
        int64_t num_layers = (Data[offset] % 4) + 1;
        offset++;
        
        bool bias = Data[offset] % 2;
        offset++;
        
        bool batch_first = Data[offset] % 2;
        offset++;
        
        double dropout = static_cast<double>(Data[offset]) / 255.0;
        offset++;
        
        bool bidirectional = Data[offset] % 2;
        offset++;
        
        // Create LSTM module
        torch::nn::LSTM lstm(
            torch::nn::LSTMOptions(input_size, hidden_size)
                .num_layers(num_layers)
                .bias(bias)
                .batch_first(batch_first)
                .dropout(dropout)
                .bidirectional(bidirectional)
        );
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape input tensor to match LSTM requirements if needed
            if (input.dim() < 2) {
                // For LSTM, we need at least 2D tensor: [seq_len, batch, input_size] or [batch, seq_len, input_size]
                if (batch_first) {
                    input = input.reshape({1, 1, input_size});
                } else {
                    input = input.reshape({1, 1, input_size});
                }
            } else if (input.dim() == 2) {
                // Add batch dimension
                if (batch_first) {
                    input = input.reshape({input.size(0), 1, input.size(1)});
                } else {
                    input = input.reshape({input.size(0), 1, input.size(1)});
                }
            }
            
            // Ensure the last dimension matches input_size
            std::vector<int64_t> new_shape = input.sizes().vec();
            new_shape[batch_first ? 2 : 2] = input_size;
            input = input.reshape(new_shape);
        } else {
            // Create a default input if we couldn't parse one
            if (batch_first) {
                input = torch::randn({2, 3, input_size});
            } else {
                input = torch::randn({2, 3, input_size});
            }
        }
        
        // Create h0 and c0 (initial hidden and cell states)
        int64_t batch_size = input.size(batch_first ? 0 : 1);
        int64_t num_directions = bidirectional ? 2 : 1;
        
        torch::Tensor h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
        torch::Tensor c0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
        
        // Forward pass
        auto output = lstm->forward(input, std::make_tuple(h0, c0));
        
        // Try to access the output
        auto output_tensor = std::get<0>(output);
        auto hidden_states = std::get<1>(output);
        
        // Test different input shapes
        torch::Tensor input2 = torch::randn({5, 2, input_size});
        if (batch_first) {
            input2 = torch::randn({2, 5, input_size});
        }
        
        auto output2 = lstm->forward(input2);
        
        // Test with different batch sizes
        torch::Tensor input3 = torch::randn({3, 4, input_size});
        if (batch_first) {
            input3 = torch::randn({4, 3, input_size});
        }
        
        auto output3 = lstm->forward(input3);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}