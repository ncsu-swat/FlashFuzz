#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input tensor has at least 3 dimensions for RNN (batch_size, seq_len, input_size)
        if (input_tensor.dim() < 3) {
            input_tensor = input_tensor.reshape({1, 1, input_tensor.numel()});
        }
        
        // Extract parameters for RNN from the remaining data
        uint8_t hidden_size = 1;
        uint8_t num_layers = 1;
        bool bidirectional = false;
        bool batch_first = false;
        float dropout = 0.0;
        
        if (offset + 4 <= Size) {
            hidden_size = Data[offset++] % 32 + 1; // Ensure non-zero
            num_layers = Data[offset++] % 3 + 1;   // 1-3 layers
            bidirectional = Data[offset++] % 2;    // 0 or 1
            batch_first = Data[offset++] % 2;      // 0 or 1
            
            if (offset + 1 <= Size) {
                dropout = static_cast<float>(Data[offset++]) / 255.0f;
            }
        }
        
        // Create scale and zero_point for quantization
        double scale = 1.0 / 256.0;
        int64_t zero_point = 0;
        
        if (offset + 2 <= Size) {
            scale = (static_cast<double>(Data[offset++]) + 1.0) / 256.0; // Ensure non-zero scale
            zero_point = static_cast<int64_t>(Data[offset++]);
        }
        
        // Create RNN options
        torch::nn::RNNOptions options(input_tensor.size(-1), hidden_size);
        options.num_layers(num_layers)
               .bias(true)
               .batch_first(batch_first)
               .dropout(dropout)
               .bidirectional(bidirectional);
        
        // Create a floating-point RNN
        torch::nn::RNN rnn_fp(options);
        
        // Create initial hidden state
        int64_t num_directions = bidirectional ? 2 : 1;
        torch::Tensor h0 = torch::zeros({num_layers * num_directions, 
                                         input_tensor.size(0), 
                                         hidden_size});
        
        // Try different operations with the RNN
        try {
            // Forward pass with hidden state
            auto output_with_h = rnn_fp->forward(input_tensor, h0);
            
            // Forward pass without hidden state
            auto output_without_h = rnn_fp->forward(input_tensor);
            
            // Access and check output properties
            auto output = std::get<0>(output_with_h);
            auto h_n = std::get<1>(output_with_h);
            
            // Test quantization on tensors
            torch::Tensor quantized_input = torch::quantize_per_tensor(
                input_tensor.to(torch::kFloat), scale, zero_point, torch::kQUInt8);
            
            // Dequantize for testing
            torch::Tensor dequantized = quantized_input.dequantize();
            
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and okay
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}