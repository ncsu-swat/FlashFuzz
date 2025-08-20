#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Parse input parameters
        uint8_t input_size = Data[offset++] % 64 + 1;
        uint8_t hidden_size = Data[offset++] % 64 + 1;
        uint8_t num_layers = (Data[offset++] % 3) + 1;
        bool bias = Data[offset++] % 2;
        bool batch_first = Data[offset++] % 2;
        double dropout = static_cast<double>(Data[offset++]) / 255.0;
        bool bidirectional = Data[offset++] % 2;
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create scale and zero_point for quantization
        double scale = 1.0 / 256.0;
        int64_t zero_point = 0;
        
        // Create regular LSTM module first
        torch::nn::LSTMOptions lstm_options = torch::nn::LSTMOptions(input_size, hidden_size)
            .num_layers(num_layers)
            .bias(bias)
            .batch_first(batch_first)
            .dropout(dropout)
            .bidirectional(bidirectional);
        
        auto lstm = torch::nn::LSTM(lstm_options);
        
        // Create initial hidden and cell states
        torch::Tensor h0, c0;
        
        // Try to create h0 and c0 if we have enough data
        if (offset < Size) {
            h0 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create default h0 with appropriate shape
            int64_t num_directions = bidirectional ? 2 : 1;
            h0 = torch::zeros({num_layers * num_directions, 1, hidden_size});
        }
        
        if (offset < Size) {
            c0 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create default c0 with appropriate shape
            int64_t num_directions = bidirectional ? 2 : 1;
            c0 = torch::zeros({num_layers * num_directions, 1, hidden_size});
        }
        
        // Quantize the input tensor
        auto q_input = torch::quantize_per_tensor(input, scale, zero_point, torch::kQUInt8);
        
        // Forward pass through the LSTM
        auto output = lstm->forward(q_input, std::make_tuple(h0, c0));
        
        // Unpack the output
        auto output_tensor = std::get<0>(output);
        auto hidden_states = std::get<1>(output);
        auto h_n = std::get<0>(hidden_states);
        auto c_n = std::get<1>(hidden_states);
        
        // Dequantize the output for further processing if needed
        auto dequantized_output = output_tensor.dequantize();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}