#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least some bytes left for parameters
        if (offset + 8 >= Size) {
            return 0;
        }
        
        // Extract parameters for RNN
        uint8_t hidden_size_byte = Data[offset++];
        int64_t hidden_size = (hidden_size_byte % 32) + 1; // Ensure positive hidden size
        
        uint8_t num_layers_byte = Data[offset++];
        int64_t num_layers = (num_layers_byte % 4) + 1; // 1-4 layers
        
        uint8_t bias_byte = Data[offset++];
        bool bias = bias_byte % 2 == 1; // 0 or 1
        
        uint8_t batch_first_byte = Data[offset++];
        bool batch_first = batch_first_byte % 2 == 1; // 0 or 1
        
        uint8_t dropout_byte = Data[offset++];
        double dropout = static_cast<double>(dropout_byte) / 255.0; // 0.0 to 1.0
        
        uint8_t bidirectional_byte = Data[offset++];
        bool bidirectional = bidirectional_byte % 2 == 1; // 0 or 1
        
        uint8_t nonlinearity_byte = Data[offset++];
        auto nonlinearity = nonlinearity_byte % 2 == 0 ? torch::kTanh : torch::kReLU;
        
        // Get input size from the input tensor
        int64_t input_size = 1;
        if (input.dim() > 0 && input.size(input.dim() - 1) > 0) {
            input_size = input.size(input.dim() - 1);
        }
        
        // Create RNN module
        torch::nn::RNNOptions options(input_size, hidden_size);
        options.num_layers(num_layers)
               .bias(bias)
               .batch_first(batch_first)
               .dropout(dropout)
               .bidirectional(bidirectional)
               .nonlinearity(nonlinearity);
        
        auto rnn = torch::nn::RNN(options);
        
        // Reshape input tensor if needed to match RNN expectations
        // RNN expects input of shape (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first=true
        if (input.dim() == 0) {
            // Convert scalar to 3D tensor
            input = input.unsqueeze(0).unsqueeze(0).unsqueeze(0);
        } else if (input.dim() == 1) {
            // Convert 1D to 3D tensor
            input = input.unsqueeze(0).unsqueeze(0);
        } else if (input.dim() == 2) {
            // Convert 2D to 3D tensor
            input = input.unsqueeze(0);
        }
        
        // Create initial hidden state
        torch::Tensor h0 = torch::zeros({num_layers * (bidirectional ? 2 : 1), 
                                         input.size(batch_first ? 0 : 1), 
                                         hidden_size});
        
        // Apply RNN
        auto output = rnn->forward(input, h0);
        
        // Access output and hidden state
        auto output_tensor = std::get<0>(output);
        auto hidden_tensor = std::get<1>(output);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output_tensor.sum() + hidden_tensor.sum();
        if (sum.item<float>() == -1.0f) {
            // This branch is unlikely to be taken, just to prevent compiler optimization
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}