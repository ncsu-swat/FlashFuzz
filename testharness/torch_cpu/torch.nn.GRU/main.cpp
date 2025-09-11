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
        
        if (Size < 10) return 0;  // Need minimum data for meaningful test
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse GRU parameters from the remaining data
        uint8_t input_size = 0;
        uint8_t hidden_size = 0;
        uint8_t num_layers = 0;
        bool bias = false;
        bool batch_first = false;
        float dropout = 0.0f;
        bool bidirectional = false;
        
        if (offset + 7 <= Size) {
            input_size = (Data[offset++] % 10) + 1;  // 1-10
            hidden_size = (Data[offset++] % 10) + 1; // 1-10
            num_layers = (Data[offset++] % 3) + 1;   // 1-3
            bias = Data[offset++] % 2;               // 0-1
            batch_first = Data[offset++] % 2;        // 0-1
            dropout = static_cast<float>(Data[offset++]) / 255.0f; // 0.0-1.0
            bidirectional = Data[offset++] % 2;      // 0-1
        }
        
        // Reshape input tensor if needed to match GRU requirements
        // GRU expects input of shape (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first=true
        if (input.dim() < 3) {
            if (input.dim() == 0) {
                input = input.reshape({1, 1, input_size});
            } else if (input.dim() == 1) {
                input = input.reshape({1, 1, input.size(0)});
            } else if (input.dim() == 2) {
                input = input.reshape({input.size(0), input.size(1), input_size});
            }
        } else if (input.dim() > 3) {
            // Flatten extra dimensions
            auto sizes = input.sizes().vec();
            int64_t seq_len = sizes[0];
            int64_t batch = sizes[1];
            int64_t features = 1;
            for (size_t i = 2; i < sizes.size(); i++) {
                features *= sizes[i];
            }
            input = input.reshape({seq_len, batch, features});
        }
        
        // Ensure the last dimension matches input_size
        auto sizes = input.sizes().vec();
        if (sizes.back() != input_size) {
            sizes.back() = input_size;
            input = input.reshape(sizes);
        }
        
        // Create GRU module
        torch::nn::GRU gru(torch::nn::GRUOptions(input_size, hidden_size)
                            .num_layers(num_layers)
                            .bias(bias)
                            .batch_first(batch_first)
                            .dropout(dropout)
                            .bidirectional(bidirectional));
        
        // Create initial hidden state (h0)
        int64_t num_directions = bidirectional ? 2 : 1;
        torch::Tensor h0;
        
        if (offset < Size) {
            h0 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape h0 to match GRU requirements: (num_layers * num_directions, batch, hidden_size)
            if (h0.dim() == 0) {
                h0 = h0.reshape({num_layers * num_directions, sizes[batch_first ? 0 : 1], hidden_size});
            } else if (h0.dim() == 1) {
                h0 = h0.reshape({num_layers * num_directions, sizes[batch_first ? 0 : 1], 
                                std::min(static_cast<int64_t>(hidden_size), h0.size(0))});
            } else if (h0.dim() == 2) {
                h0 = h0.reshape({num_layers * num_directions, h0.size(0), 
                                std::min(static_cast<int64_t>(hidden_size), h0.size(1))});
            } else {
                auto h0_sizes = h0.sizes().vec();
                h0_sizes[0] = num_layers * num_directions;
                h0_sizes[1] = sizes[batch_first ? 0 : 1];
                h0_sizes[2] = hidden_size;
                h0 = h0.reshape(h0_sizes);
            }
        } else {
            // Create default h0 if not enough data
            h0 = torch::zeros({num_layers * num_directions, sizes[batch_first ? 0 : 1], hidden_size});
        }
        
        // Convert tensors to same dtype
        if (h0.dtype() != input.dtype()) {
            h0 = h0.to(input.dtype());
        }
        
        // Forward pass
        auto output = gru->forward(input, h0);
        
        // Get output and hidden state
        auto output_tensor = std::get<0>(output);
        auto hidden_state = std::get<1>(output);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output_tensor.sum() + hidden_state.sum();
        if (sum.isnan().item<bool>()) {
            throw std::runtime_error("NaN detected in GRU output");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
