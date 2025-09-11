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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for RNN
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        int64_t num_layers = 0;
        bool bias = false;
        bool batch_first = false;
        double dropout = 0.0;
        bool bidirectional = false;
        
        if (offset + 8 <= Size) {
            std::memcpy(&input_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            input_size = std::abs(input_size) % 32 + 1;
        } else {
            input_size = 10;
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&hidden_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            hidden_size = std::abs(hidden_size) % 32 + 1;
        } else {
            hidden_size = 20;
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&num_layers, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_layers = std::abs(num_layers) % 3 + 1;
        } else {
            num_layers = 1;
        }
        
        if (offset < Size) {
            bias = Data[offset++] & 1;
        }
        
        if (offset < Size) {
            batch_first = Data[offset++] & 1;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&dropout, Data + offset, sizeof(double));
            offset += sizeof(double);
            dropout = std::abs(dropout) / 10.0;
        }
        
        if (offset < Size) {
            bidirectional = Data[offset++] & 1;
        }
        
        // Reshape input tensor if needed to match expected RNN input shape
        auto input_sizes = input_tensor.sizes().vec();
        if (input_sizes.size() < 2) {
            // For RNN, we need at least 2D tensor (seq_len, batch, input_size) or (batch, seq_len, input_size)
            if (input_sizes.size() == 0) {
                input_tensor = input_tensor.reshape({1, 1});
            } else if (input_sizes.size() == 1) {
                input_tensor = input_tensor.reshape({input_sizes[0], 1});
            }
            input_sizes = input_tensor.sizes().vec();
        }
        
        // Ensure the last dimension is input_size
        if (input_sizes.size() >= 3) {
            // For 3D tensor, reshape to ensure last dimension is input_size
            int64_t seq_len = input_sizes[batch_first ? 1 : 0];
            int64_t batch = input_sizes[batch_first ? 0 : 1];
            input_tensor = input_tensor.reshape({batch_first ? batch : seq_len, batch_first ? seq_len : batch, input_size});
        } else if (input_sizes.size() == 2) {
            // For 2D tensor, add input_size dimension
            int64_t dim0 = input_sizes[0];
            int64_t dim1 = input_sizes[1];
            input_tensor = input_tensor.reshape({dim0, dim1, input_size});
        }
        
        // Create initial hidden state
        torch::Tensor h0 = torch::zeros({num_layers * (bidirectional ? 2 : 1), 
                                         batch_first ? input_tensor.size(0) : input_tensor.size(1), 
                                         hidden_size});
        
        // Create the RNN module using regular torch::nn::RNN
        torch::nn::RNNOptions options(input_size, hidden_size);
        options.num_layers(num_layers)
               .bias(bias)
               .batch_first(batch_first)
               .dropout(dropout)
               .bidirectional(bidirectional);
        
        torch::nn::RNN rnn_module(options);
        
        // Forward pass
        auto output = rnn_module->forward(input_tensor, h0);
        
        // Get output tensor and final hidden state
        auto output_tensor = std::get<0>(output);
        auto h_n = std::get<1>(output);
        
        // Test some operations on the output to ensure it's valid
        auto sum = output_tensor.sum();
        auto mean = output_tensor.mean();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
