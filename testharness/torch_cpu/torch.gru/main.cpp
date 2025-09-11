#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 10) return 0; // Need minimum data for basic parameters
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions for GRU (batch_size, seq_len, input_size)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract dimensions from input
        int64_t batch_size = input.size(0);
        int64_t seq_len = input.size(1);
        int64_t input_size = input.size(2);
        
        // Create hidden state tensor
        int64_t hidden_size = 0;
        int64_t num_layers = 1;
        
        // Extract parameters from remaining data
        if (offset + 8 <= Size) {
            std::memcpy(&hidden_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure hidden_size is reasonable
            hidden_size = std::abs(hidden_size) % 64 + 1;
        } else {
            hidden_size = 4; // Default value
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&num_layers, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure num_layers is reasonable
            num_layers = std::abs(num_layers) % 3 + 1;
        }
        
        // Create a hidden state tensor
        torch::Tensor h0 = torch::zeros({num_layers, batch_size, hidden_size}, input.options());
        
        // Create GRU options
        bool bias = true;
        bool batch_first = true;
        bool bidirectional = false;
        double dropout = 0.0;
        
        // Extract more parameters if data available
        if (offset + 3 <= Size) {
            bias = Data[offset++] & 1;
            batch_first = Data[offset++] & 1;
            bidirectional = Data[offset++] & 1;
        }
        
        if (offset + 8 <= Size) {
            double dropout_raw;
            std::memcpy(&dropout_raw, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure dropout is between 0 and 1
            dropout = std::abs(dropout_raw) / (1.0 + std::abs(dropout_raw));
        }
        
        // Create GRU module
        torch::nn::GRUOptions options = torch::nn::GRUOptions(input_size, hidden_size)
            .num_layers(num_layers)
            .bias(bias)
            .batch_first(batch_first)
            .bidirectional(bidirectional)
            .dropout(dropout);
        
        torch::nn::GRU gru_module(options);
        
        // Convert module to the same dtype as input
        gru_module->to(input.scalar_type());
        
        // Run GRU
        auto output = gru_module->forward(input, h0);
        
        // Access output tensors
        torch::Tensor output_seq = std::get<0>(output);
        torch::Tensor h_n = std::get<1>(output);
        
        // Perform some operations on the output to ensure it's used
        auto sum1 = output_seq.sum();
        auto sum2 = h_n.sum();
        auto total_sum = sum1 + sum2;
        
        // Try different input configurations if there's enough data
        if (offset + 10 < Size) {
            // Try with a different batch size
            int64_t new_batch_size = std::max(int64_t(1), batch_size / 2);
            torch::Tensor input2 = input.slice(0, 0, new_batch_size);
            torch::Tensor h02 = h0.slice(1, 0, new_batch_size);
            
            auto output2 = gru_module->forward(input2, h02);
        }
        
        // Try with no initial hidden state
        auto output_no_h0 = gru_module->forward(input);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
