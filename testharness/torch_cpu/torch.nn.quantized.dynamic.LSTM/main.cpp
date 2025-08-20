#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 8) {
            return 0;
        }
        
        // Parse input parameters for LSTM
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
            input_size = std::abs(input_size) % 128 + 1; // Ensure positive and reasonable size
        } else {
            input_size = 10; // Default
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&hidden_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            hidden_size = std::abs(hidden_size) % 128 + 1; // Ensure positive and reasonable size
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
            bias = Data[offset++] & 0x1; // Use lowest bit for boolean
        }
        
        if (offset < Size) {
            batch_first = Data[offset++] & 0x1; // Use lowest bit for boolean
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&dropout, Data + offset, sizeof(double));
            offset += sizeof(double);
            dropout = std::abs(dropout) / 10.0; // Ensure reasonable dropout value
            if (dropout > 0.9) dropout = 0.9;
        }
        
        if (offset < Size) {
            bidirectional = Data[offset++] & 0x1; // Use lowest bit for boolean
        }
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            try {
                input = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Reshape input to match LSTM requirements if needed
                if (input.dim() < 2) {
                    // For LSTM, we need at least 2D tensor (seq_len, input_size)
                    if (batch_first) {
                        input = input.reshape({1, 1, input_size});
                    } else {
                        input = input.reshape({1, 1, input_size});
                    }
                } else if (input.dim() == 2) {
                    // Add batch dimension
                    if (batch_first) {
                        input = input.reshape({1, input.size(0), input.size(1)});
                    } else {
                        input = input.reshape({input.size(0), 1, input.size(1)});
                    }
                }
                
                // Ensure last dimension matches input_size
                std::vector<int64_t> new_shape = input.sizes().vec();
                new_shape[2] = input_size;
                input = input.reshape(new_shape);
                
            } catch (const std::exception& e) {
                // If tensor creation fails, create a simple one
                if (batch_first) {
                    input = torch::randn({2, 3, input_size});
                } else {
                    input = torch::randn({3, 2, input_size});
                }
            }
        } else {
            // Default input if not enough data
            if (batch_first) {
                input = torch::randn({2, 3, input_size});
            } else {
                input = torch::randn({3, 2, input_size});
            }
        }
        
        // Create regular LSTM and use quantized dynamic operations
        torch::nn::LSTMOptions options(input_size, hidden_size);
        options.num_layers(num_layers)
               .bias(bias)
               .batch_first(batch_first)
               .dropout(dropout)
               .bidirectional(bidirectional);
        
        auto lstm = torch::nn::LSTM(options);
        
        // Create initial hidden state (h0, c0)
        // h0 shape: (num_layers * num_directions, batch_size, hidden_size)
        // c0 shape: (num_layers * num_directions, batch_size, hidden_size)
        int64_t num_directions = bidirectional ? 2 : 1;
        int64_t batch_size = batch_first ? input.size(0) : input.size(1);
        
        torch::Tensor h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
        torch::Tensor c0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
        
        // Run LSTM forward
        auto output = lstm->forward(input, std::make_tuple(h0, c0));
        
        // Extract output and hidden state
        auto output_tensor = std::get<0>(output);
        auto hidden_state = std::get<1>(output);
        
        // Access h_n and c_n from hidden_state
        auto h_n = std::get<0>(hidden_state);
        auto c_n = std::get<1>(hidden_state);
        
        // Apply quantized dynamic operations on the output
        auto quantized_output = torch::quantize_per_tensor(output_tensor, 0.1, 128, torch::kQUInt8);
        auto dequantized_output = quantized_output.dequantize();
        
        // Perform some operations on the output to ensure it's used
        auto sum = dequantized_output.sum() + h_n.sum() + c_n.sum();
        if (sum.item<float>() == -1000000.0f) {
            // This condition is unlikely to be true, just to prevent compiler optimization
            std::cerr << "Unexpected sum value" << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}