#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Early exit if not enough data
        if (Size < 10) {
            return 0;
        }
        
        // Parse input parameters for GRU
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        int64_t num_layers = 0;
        bool bias = false;
        bool batch_first = false;
        bool bidirectional = false;
        float dropout = 0.0;
        
        // Extract parameters from input data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&input_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            input_size = std::abs(input_size) % 100 + 1; // Ensure positive and reasonable size
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&hidden_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            hidden_size = std::abs(hidden_size) % 100 + 1; // Ensure positive and reasonable size
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&num_layers, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_layers = std::abs(num_layers) % 3 + 1; // Limit to 1-3 layers for efficiency
        }
        
        if (offset < Size) {
            bias = Data[offset++] & 1; // Use lowest bit to determine boolean
        }
        
        if (offset < Size) {
            batch_first = Data[offset++] & 1;
        }
        
        if (offset < Size) {
            bidirectional = Data[offset++] & 1;
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&dropout, Data + offset, sizeof(float));
            offset += sizeof(float);
            dropout = std::fabs(dropout) / 10.0f; // Ensure positive and reasonable dropout rate
        }
        
        // Create the GRU module using regular GRU (quantized dynamic GRU may not be available in C++ frontend)
        torch::nn::GRUOptions options(input_size, hidden_size);
        options.num_layers(num_layers)
               .bias(bias)
               .batch_first(batch_first)
               .dropout(dropout)
               .bidirectional(bidirectional);
        
        auto gru = torch::nn::GRU(options);
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape input tensor to match GRU requirements if needed
            if (input.dim() < 2) {
                // For 0D or 1D tensors, reshape to 2D
                if (batch_first) {
                    input = input.reshape({1, -1});
                } else {
                    input = input.reshape({-1, 1});
                }
            }
            
            if (input.dim() == 2) {
                // Add sequence dimension if missing
                if (batch_first) {
                    input = input.unsqueeze(1); // [batch, 1, features]
                } else {
                    input = input.unsqueeze(0); // [1, batch, features]
                }
            }
            
            // Ensure the feature dimension matches input_size
            int64_t feature_dim = batch_first ? 2 : 2;
            if (input.size(feature_dim) != input_size) {
                std::vector<int64_t> new_shape = input.sizes().vec();
                new_shape[feature_dim] = input_size;
                input = input.reshape(new_shape);
            }
        } else {
            // Create a default input tensor if we couldn't parse one
            if (batch_first) {
                input = torch::randn({2, 3, input_size}); // [batch, seq, features]
            } else {
                input = torch::randn({3, 2, input_size}); // [seq, batch, features]
            }
        }
        
        // Create h0 (initial hidden state)
        torch::Tensor h0;
        if (offset < Size) {
            h0 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape h0 to match GRU requirements
            int64_t num_directions = bidirectional ? 2 : 1;
            std::vector<int64_t> h0_shape = {num_layers * num_directions, input.size(batch_first ? 0 : 1), hidden_size};
            
            if (h0.numel() > 0) {
                // Try to reshape if possible
                h0 = h0.reshape(h0_shape);
            } else {
                // Create a new tensor if empty
                h0 = torch::zeros(h0_shape);
            }
        } else {
            // Create a default h0 if we couldn't parse one
            int64_t num_directions = bidirectional ? 2 : 1;
            int64_t batch_size = batch_first ? input.size(0) : input.size(1);
            h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
        }
        
        // Run the GRU
        auto output = gru->forward(input, h0);
        
        // Access the output to ensure computation is not optimized away
        auto output_tensor = std::get<0>(output);
        auto h_n = std::get<1>(output);
        
        // Force evaluation
        output_tensor.sum().item<float>();
        h_n.sum().item<float>();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}