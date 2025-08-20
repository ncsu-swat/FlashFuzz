#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) return 0;
        
        // Parse input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions for GRU (batch_size, seq_len, input_size)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract dimensions from input
        int64_t batch_size = input.size(0);
        int64_t seq_len = input.size(1);
        int64_t input_size = input.size(2);
        
        // Parse parameters for GRU
        int64_t hidden_size = 0;
        int64_t num_layers = 1;
        bool bias = true;
        bool batch_first = false;
        bool bidirectional = false;
        double dropout = 0.0;
        
        // Extract parameters from remaining data if available
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&hidden_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            hidden_size = std::abs(hidden_size) % 64 + 1; // Ensure positive and reasonable size
        } else {
            hidden_size = input_size; // Default to input size
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&num_layers, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_layers = std::abs(num_layers) % 3 + 1; // 1-3 layers
        }
        
        if (offset < Size) {
            bias = Data[offset++] & 1; // Use lowest bit
        }
        
        if (offset < Size) {
            batch_first = Data[offset++] & 1; // Use lowest bit
        }
        
        if (offset < Size) {
            bidirectional = Data[offset++] & 1; // Use lowest bit
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&dropout, Data + offset, sizeof(double));
            offset += sizeof(double);
            dropout = std::abs(dropout) / 10.0; // Ensure reasonable dropout value
        }
        
        // Create a regular GRU module since quantized dynamic GRU is not available in C++ frontend
        torch::nn::GRUOptions options = 
            torch::nn::GRUOptions(input_size, hidden_size)
                .num_layers(num_layers)
                .bias(bias)
                .batch_first(batch_first)
                .bidirectional(bidirectional)
                .dropout(dropout);
        
        torch::nn::GRU gru_module(options);
        
        // Create initial hidden state (optional)
        torch::Tensor h0;
        bool use_h0 = false;
        
        if (offset < Size) {
            use_h0 = Data[offset++] & 1;
        }
        
        if (use_h0) {
            int64_t num_directions = bidirectional ? 2 : 1;
            h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
            
            // Fill h0 with some values if we have more data
            if (offset < Size) {
                h0.fill_(static_cast<float>(Data[offset++]) / 255.0f);
            }
        }
        
        // Forward pass
        torch::Tensor output;
        torch::Tensor h_n;
        
        if (use_h0) {
            auto result = gru_module->forward(input, h0);
            output = std::get<0>(result);
            h_n = std::get<1>(result);
        } else {
            auto result = gru_module->forward(input);
            output = std::get<0>(result);
            h_n = std::get<1>(result);
        }
        
        // Test other methods
        if (offset < Size && Data[offset++] % 2 == 0) {
            gru_module->to(torch::kCPU);
        }
        
        // Test parameter access
        if (offset < Size && Data[offset++] % 3 == 0) {
            auto parameters = gru_module->parameters();
        }
        
        if (offset < Size && Data[offset++] % 3 == 1) {
            auto named_parameters = gru_module->named_parameters();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}