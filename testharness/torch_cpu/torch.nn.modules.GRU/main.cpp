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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 4 more bytes for GRU parameters
        if (Size - offset < 4) {
            return 0;
        }
        
        // Parse GRU parameters
        uint8_t input_size_byte = Data[offset++];
        uint8_t hidden_size_byte = Data[offset++];
        uint8_t num_layers_byte = Data[offset++];
        uint8_t flags_byte = Data[offset++];
        
        // Convert to usable parameters with reasonable bounds
        int64_t input_size = (input_size_byte % 32) + 1;  // 1-32
        int64_t hidden_size = (hidden_size_byte % 32) + 1;  // 1-32
        int64_t num_layers = (num_layers_byte % 3) + 1;  // 1-3
        
        // Extract boolean flags
        bool batch_first = (flags_byte & 0x01) != 0;
        bool bidirectional = (flags_byte & 0x02) != 0;
        bool dropout_flag = (flags_byte & 0x04) != 0;
        
        // Set dropout value if enabled
        double dropout = 0.0;
        if (dropout_flag && Size - offset >= 1) {
            uint8_t dropout_byte = Data[offset++];
            dropout = static_cast<double>(dropout_byte) / 255.0;  // 0.0-1.0
        }
        
        // Create GRU module
        torch::nn::GRU gru_options = torch::nn::GRU(
            torch::nn::GRUOptions(input_size, hidden_size)
                .num_layers(num_layers)
                .batch_first(batch_first)
                .bidirectional(bidirectional)
                .dropout(dropout)
        );
        
        // Reshape input tensor if needed to match GRU expectations
        // GRU expects input of shape (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first=true
        if (input.dim() == 0) {
            // Scalar tensor - reshape to 3D
            input = input.reshape({1, 1, 1});
        } else if (input.dim() == 1) {
            // 1D tensor - reshape to 3D
            input = input.reshape({1, 1, input.size(0)});
        } else if (input.dim() == 2) {
            // 2D tensor - add batch dimension
            input = input.unsqueeze(0);
        }
        
        // Ensure the last dimension matches input_size
        auto input_sizes = input.sizes().vec();
        if (input.dim() >= 3) {
            int64_t last_dim_idx = batch_first ? 2 : 1;
            if (input_sizes[last_dim_idx] != input_size) {
                // Resize the last dimension to match input_size
                input_sizes[last_dim_idx] = input_size;
                input = input.resize_(input_sizes);
            }
        }
        
        // Create initial hidden state (optional)
        torch::Tensor h0;
        bool use_h0 = false;
        
        if (Size - offset > 10) {
            use_h0 = true;
            h0 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape h0 to match GRU expectations: (num_layers * num_directions, batch, hidden_size)
            int64_t num_directions = bidirectional ? 2 : 1;
            int64_t batch_size = batch_first ? input_sizes[0] : input_sizes[1];
            
            if (h0.dim() == 0) {
                h0 = h0.reshape({num_layers * num_directions, batch_size, hidden_size});
            } else if (h0.dim() == 1) {
                h0 = h0.reshape({num_layers * num_directions, batch_size, 1});
                h0 = h0.expand({num_layers * num_directions, batch_size, hidden_size});
            } else if (h0.dim() == 2) {
                h0 = h0.unsqueeze(0);
                h0 = h0.expand({num_layers * num_directions, h0.size(1), h0.size(2)});
            } else {
                // Ensure h0 has the right shape
                auto h0_sizes = h0.sizes().vec();
                h0_sizes[0] = num_layers * num_directions;
                if (h0.dim() > 1) {
                    h0_sizes[1] = batch_size;
                }
                if (h0.dim() > 2) {
                    h0_sizes[2] = hidden_size;
                }
                h0 = h0.resize_(h0_sizes);
            }
        }
        
        // Apply GRU
        std::tuple<torch::Tensor, torch::Tensor> output;
        if (use_h0) {
            output = gru_options->forward(input, h0);
        } else {
            output = gru_options->forward(input);
        }
        
        // Access output tensors
        torch::Tensor output_seq = std::get<0>(output);
        torch::Tensor h_n = std::get<1>(output);
        
        // Perform some operations on the output to ensure it's used
        auto sum1 = output_seq.sum();
        auto sum2 = h_n.sum();
        auto total_sum = sum1 + sum2;
        
        // Prevent compiler from optimizing away the computation
        if (total_sum.item<float>() == -12345.6789f) {
            throw std::runtime_error("This should never happen");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
