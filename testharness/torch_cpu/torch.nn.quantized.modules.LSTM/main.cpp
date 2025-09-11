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
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Parse parameters for LSTM
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        int64_t num_layers = 0;
        bool bias = false;
        bool batch_first = false;
        double dropout = 0.0;
        bool bidirectional = false;
        
        // Extract parameters from input data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&input_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            input_size = std::abs(input_size) % 100 + 1; // Ensure positive and reasonable size
        } else {
            input_size = 10; // Default
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&hidden_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            hidden_size = std::abs(hidden_size) % 100 + 1; // Ensure positive and reasonable size
        } else {
            hidden_size = 20; // Default
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&num_layers, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_layers = std::abs(num_layers) % 3 + 1; // 1-3 layers is reasonable
        } else {
            num_layers = 1; // Default
        }
        
        if (offset < Size) {
            bias = Data[offset++] & 1; // Use lowest bit to determine boolean
        }
        
        if (offset < Size) {
            batch_first = Data[offset++] & 1;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&dropout, Data + offset, sizeof(double));
            offset += sizeof(double);
            dropout = std::abs(dropout) / 10.0; // Scale to reasonable dropout value
            if (dropout > 0.9) dropout = 0.9;
        }
        
        if (offset < Size) {
            bidirectional = Data[offset++] & 1;
        }
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape input to match LSTM requirements if needed
            // LSTM expects [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
            std::vector<int64_t> input_shape;
            if (input.dim() == 0) {
                // Scalar tensor - reshape to 3D
                input = input.reshape({1, 1, input_size});
            } else if (input.dim() == 1) {
                // 1D tensor - reshape to 3D
                int64_t seq_len = input.size(0) > 0 ? input.size(0) : 1;
                input = input.reshape({seq_len, 1, input_size});
            } else if (input.dim() == 2) {
                // 2D tensor - add batch dimension
                int64_t seq_len = input.size(0) > 0 ? input.size(0) : 1;
                int64_t batch = input.size(1) > 0 ? input.size(1) : 1;
                input = input.reshape({seq_len, batch, input_size});
            } else if (input.dim() >= 3) {
                // 3D+ tensor - reshape to ensure last dimension is input_size
                std::vector<int64_t> new_shape;
                if (batch_first) {
                    new_shape = {input.size(0) > 0 ? input.size(0) : 1, 
                                input.size(1) > 0 ? input.size(1) : 1, 
                                input_size};
                } else {
                    new_shape = {input.size(0) > 0 ? input.size(0) : 1, 
                                input.size(1) > 0 ? input.size(1) : 1, 
                                input_size};
                }
                input = input.reshape(new_shape);
            }
        } else {
            // Create a default input tensor if we don't have enough data
            if (batch_first) {
                input = torch::randn({2, 3, input_size});
            } else {
                input = torch::randn({3, 2, input_size});
            }
        }
        
        // Ensure input has float data type for quantization
        if (input.scalar_type() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Create regular LSTM module first, then quantize it
        torch::nn::LSTMOptions options(input_size, hidden_size);
        options.num_layers(num_layers)
               .bias(bias)
               .batch_first(batch_first)
               .dropout(dropout)
               .bidirectional(bidirectional);
        
        // Create a regular LSTM module
        torch::nn::LSTM lstm(options);
        
        // Create initial hidden state (h0, c0)
        int64_t num_directions = bidirectional ? 2 : 1;
        torch::Tensor h0 = torch::zeros({num_layers * num_directions, 
                                         batch_first ? input.size(0) : input.size(1), 
                                         hidden_size});
        torch::Tensor c0 = torch::zeros({num_layers * num_directions, 
                                         batch_first ? input.size(0) : input.size(1), 
                                         hidden_size});
        
        // Forward pass
        auto output = lstm->forward(input, std::make_tuple(h0, c0));
        
        // Extract output and hidden states
        auto output_tensor = std::get<0>(output);
        auto hidden_states = std::get<1>(output);
        
        // Access h_n and c_n from hidden_states
        auto h_n = std::get<0>(hidden_states);
        auto c_n = std::get<1>(hidden_states);
        
        // Perform some operations on the outputs to ensure they're used
        auto sum_output = output_tensor.sum();
        auto sum_h_n = h_n.sum();
        auto sum_c_n = c_n.sum();
        
        // Prevent compiler from optimizing away the computations
        volatile float dummy = sum_output.item<float>() + sum_h_n.item<float>() + sum_c_n.item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
