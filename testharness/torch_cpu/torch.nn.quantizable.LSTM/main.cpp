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
        
        // Parse LSTM parameters from the remaining data
        int64_t input_size = 10;
        int64_t hidden_size = 20;
        int64_t num_layers = 1;
        bool bias = true;
        bool batch_first = false;
        double dropout = 0.0;
        bool bidirectional = false;
        
        if (offset + 8 <= Size) {
            input_size = static_cast<int64_t>(Data[offset++]) % 100 + 1;
            hidden_size = static_cast<int64_t>(Data[offset++]) % 100 + 1;
            num_layers = static_cast<int64_t>(Data[offset++]) % 4 + 1;
            bias = Data[offset++] % 2 == 0;
            batch_first = Data[offset++] % 2 == 0;
            dropout = static_cast<double>(Data[offset++]) / 255.0;
            bidirectional = Data[offset++] % 2 == 0;
        }
        
        // Reshape input tensor if needed to match LSTM requirements
        // LSTM expects input of shape (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first=true
        if (input_tensor.dim() < 3) {
            if (input_tensor.dim() == 0) {
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0);
            } else if (input_tensor.dim() == 1) {
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0);
            } else if (input_tensor.dim() == 2) {
                input_tensor = input_tensor.unsqueeze(0);
            }
        }
        
        // Ensure the last dimension is input_size
        auto sizes = input_tensor.sizes().vec();
        if (sizes.size() >= 3) {
            sizes[sizes.size() - 1] = input_size;
            input_tensor = input_tensor.reshape(sizes);
        }
        
        // Convert to float if not already a floating point type
        if (!torch::isFloatingType(input_tensor.scalar_type())) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Create LSTM module
        torch::nn::LSTM lstm(torch::nn::LSTMOptions(input_size, hidden_size)
                                .num_layers(num_layers)
                                .bias(bias)
                                .batch_first(batch_first)
                                .dropout(dropout)
                                .bidirectional(bidirectional));
        
        // Create initial hidden state and cell state
        int64_t batch_size = batch_first ? sizes[0] : sizes[1];
        int64_t directions = bidirectional ? 2 : 1;
        
        torch::Tensor h0 = torch::zeros({num_layers * directions, batch_size, hidden_size});
        torch::Tensor c0 = torch::zeros({num_layers * directions, batch_size, hidden_size});
        
        // Forward pass
        auto output = lstm->forward(input_tensor, std::make_tuple(h0, c0));
        
        // Get output tensor and final states
        torch::Tensor output_tensor = std::get<0>(output);
        auto states = std::get<1>(output);
        torch::Tensor h_n = std::get<0>(states);
        torch::Tensor c_n = std::get<1>(states);
        
        // Simple quantization simulation by converting to int8 and back
        if (offset < Size) {
            bool quantize = Data[offset++] % 2 == 0;
            if (quantize) {
                // Simulate quantization by converting tensors to int8 and back
                auto quantized_input = input_tensor.to(torch::kInt8).to(torch::kFloat);
                auto quantized_h0 = h0.to(torch::kInt8).to(torch::kFloat);
                auto quantized_c0 = c0.to(torch::kInt8).to(torch::kFloat);
                
                // Run inference with the "quantized" inputs
                auto quantized_output = lstm->forward(quantized_input, std::make_tuple(quantized_h0, quantized_c0));
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
