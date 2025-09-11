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
        
        bool bidirectional = Data[offset] % 2;
        offset++;
        
        float dropout = static_cast<float>(Data[offset]) / 255.0f;
        offset++;
        
        // Create the LSTM module (using regular LSTM since quantized dynamic LSTM is not available)
        torch::nn::LSTMOptions options(input_size, hidden_size);
        options.num_layers(num_layers)
               .bias(bias)
               .batch_first(batch_first)
               .dropout(dropout)
               .bidirectional(bidirectional);
        
        auto lstm = torch::nn::LSTM(options);
        
        // Create input tensor
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try with a simple tensor
            int64_t batch_size = 3;
            int64_t seq_length = 4;
            
            if (batch_first) {
                input = torch::randn({batch_size, seq_length, input_size});
            } else {
                input = torch::randn({seq_length, batch_size, input_size});
            }
        }
        
        // Reshape input tensor if needed to match LSTM requirements
        std::vector<int64_t> input_sizes = input.sizes().vec();
        if (input_sizes.size() < 2) {
            // Add dimensions to make it at least 2D
            while (input_sizes.size() < 2) {
                input_sizes.push_back(1);
            }
            
            // Ensure the last dimension matches input_size
            input_sizes.back() = input_size;
            input = input.reshape(input_sizes);
        } else if (input_sizes.size() > 3) {
            // Truncate to 3D
            input_sizes.resize(3);
            input = input.reshape(input_sizes);
        }
        
        // Ensure the feature dimension matches input_size
        int feature_dim = batch_first ? 2 : 1;
        if (input_sizes.size() > feature_dim && input_sizes[feature_dim] != input_size) {
            input_sizes[feature_dim] = input_size;
            try {
                input = input.reshape(input_sizes);
            } catch (const std::exception& e) {
                // If reshape fails, create a new tensor
                if (batch_first) {
                    input = torch::randn({input_sizes[0], input_sizes[1], input_size});
                } else {
                    input = torch::randn({input_sizes[0], input_sizes[1], input_size});
                }
            }
        }
        
        // Create h0 and c0 (initial hidden and cell states)
        torch::Tensor h0, c0;
        bool use_initial_states = false;
        
        if (offset + 1 < Size && Data[offset] % 2 == 1) {
            use_initial_states = true;
            offset++;
            
            int64_t batch_dim = batch_first ? 0 : 1;
            int64_t batch_size = input.size(batch_dim);
            int64_t num_directions = bidirectional ? 2 : 1;
            
            try {
                h0 = fuzzer_utils::createTensor(Data, Size, offset);
                h0 = h0.reshape({num_layers * num_directions, batch_size, hidden_size});
            } catch (const std::exception& e) {
                h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
            }
            
            try {
                c0 = fuzzer_utils::createTensor(Data, Size, offset);
                c0 = c0.reshape({num_layers * num_directions, batch_size, hidden_size});
            } catch (const std::exception& e) {
                c0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
            }
        }
        
        // Forward pass
        torch::Tensor output;
        torch::Tensor h_n;
        torch::Tensor c_n;
        
        if (use_initial_states) {
            auto result = lstm->forward(input, std::make_tuple(h0, c0));
            output = std::get<0>(result);
            h_n = std::get<0>(std::get<1>(result));
            c_n = std::get<1>(std::get<1>(result));
        } else {
            auto result = lstm->forward(input);
            output = std::get<0>(result);
            h_n = std::get<0>(std::get<1>(result));
            c_n = std::get<1>(std::get<1>(result));
        }
        
        // Test some operations on the output to ensure it's valid
        auto output_sum = output.sum();
        auto h_n_sum = h_n.sum();
        auto c_n_sum = c_n.sum();
        
        // Test state_dict and load_state_dict
        auto state_dict = lstm->state_dict();
        lstm->load_state_dict(state_dict);
        
        // Test to(device)
        lstm->to(torch::kCPU);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
