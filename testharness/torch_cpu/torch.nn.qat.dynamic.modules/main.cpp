#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a dynamic quantized linear module
        int64_t in_features = 0;
        int64_t out_features = 0;
        
        // Extract in_features and out_features from the input data if available
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&in_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            in_features = std::abs(in_features) % 128 + 1; // Ensure positive and reasonable size
        } else {
            in_features = 10; // Default value
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_features = std::abs(out_features) % 128 + 1; // Ensure positive and reasonable size
        } else {
            out_features = 5; // Default value
        }
        
        // Create a regular linear module (QAT modules may not be available in C++ frontend)
        torch::nn::LinearOptions linear_options(in_features, out_features);
        
        // Extract bias option from input data if available
        bool with_bias = true;
        if (offset < Size) {
            with_bias = Data[offset++] & 0x1; // Use lowest bit to determine bias
        }
        linear_options.bias(with_bias);
        
        // Create the linear module
        torch::nn::Linear linear_module(linear_options);
        
        // Reshape input tensor if needed to match the expected input shape for the linear layer
        if (input_tensor.dim() == 0) {
            // For scalar tensor, reshape to [1, in_features]
            input_tensor = input_tensor.reshape({1, in_features});
        } else if (input_tensor.dim() == 1) {
            // For 1D tensor, reshape to [1, min(size, in_features)]
            int64_t size = input_tensor.size(0);
            if (size < in_features) {
                input_tensor = input_tensor.reshape({1, size});
                // Pad if necessary
                input_tensor = torch::nn::functional::pad(input_tensor, 
                    torch::nn::functional::PadFuncOptions({0, in_features - size}));
            } else {
                input_tensor = input_tensor.slice(0, 0, in_features).reshape({1, in_features});
            }
        } else {
            // For multi-dimensional tensor, ensure the last dimension is in_features
            std::vector<int64_t> new_shape = input_tensor.sizes().vec();
            if (new_shape.size() >= 2) {
                new_shape[new_shape.size() - 1] = in_features;
                try {
                    input_tensor = input_tensor.reshape(new_shape);
                } catch (...) {
                    // If reshape fails, create a new tensor with the right shape
                    input_tensor = torch::ones(new_shape, input_tensor.options());
                }
            } else {
                // If tensor doesn't have at least 2 dimensions, reshape it
                input_tensor = input_tensor.reshape({1, in_features});
            }
        }
        
        // Apply the module to the input tensor
        torch::Tensor output;
        try {
            output = linear_module->forward(input_tensor);
        } catch (...) {
            // If forward fails, try with a properly shaped tensor
            input_tensor = torch::ones({1, in_features}, torch::kFloat);
            output = linear_module->forward(input_tensor);
        }
        
        // Test other modules if we have enough data
        if (offset + 4 < Size) {
            // Extract parameters for Conv2d
            int64_t in_channels = std::abs(static_cast<int64_t>(Data[offset++])) % 16 + 1;
            int64_t out_channels = std::abs(static_cast<int64_t>(Data[offset++])) % 16 + 1;
            int64_t kernel_size = std::abs(static_cast<int64_t>(Data[offset++])) % 7 + 1;
            
            // Create a Conv2d module
            torch::nn::Conv2dOptions conv_options(in_channels, out_channels, kernel_size);
            
            // Extract padding, stride, dilation from input data
            if (offset < Size) {
                int64_t padding = std::abs(static_cast<int64_t>(Data[offset++])) % 3;
                conv_options.padding(padding);
            }
            
            if (offset < Size) {
                int64_t stride = std::abs(static_cast<int64_t>(Data[offset++])) % 3 + 1;
                conv_options.stride(stride);
            }
            
            if (offset < Size) {
                int64_t dilation = std::abs(static_cast<int64_t>(Data[offset++])) % 2 + 1;
                conv_options.dilation(dilation);
            }
            
            if (offset < Size) {
                bool with_bias = Data[offset++] & 0x1;
                conv_options.bias(with_bias);
            }
            
            // Create the Conv2d module
            torch::nn::Conv2d conv_module(conv_options);
            
            // Create a properly shaped input tensor for Conv2d
            torch::Tensor conv_input = torch::ones({1, in_channels, 28, 28}, torch::kFloat);
            
            // Apply the module
            torch::Tensor conv_output = conv_module->forward(conv_input);
        }
        
        // Test LSTM if we have enough data
        if (offset + 4 < Size) {
            int64_t input_size = std::abs(static_cast<int64_t>(Data[offset++])) % 32 + 1;
            int64_t hidden_size = std::abs(static_cast<int64_t>(Data[offset++])) % 32 + 1;
            int64_t num_layers = std::abs(static_cast<int64_t>(Data[offset++])) % 3 + 1;
            
            torch::nn::LSTMOptions lstm_options(input_size, hidden_size);
            lstm_options.num_layers(num_layers);
            
            if (offset < Size) {
                bool bidirectional = Data[offset++] & 0x1;
                lstm_options.bidirectional(bidirectional);
            }
            
            if (offset < Size) {
                bool batch_first = Data[offset++] & 0x1;
                lstm_options.batch_first(batch_first);
            }
            
            if (offset < Size) {
                double dropout = static_cast<double>(Data[offset++]) / 255.0;
                lstm_options.dropout(dropout);
            }
            
            // Create the LSTM module
            torch::nn::LSTM lstm_module(lstm_options);
            
            // Create a properly shaped input tensor for LSTM
            int64_t seq_len = 10;
            int64_t batch_size = 3;
            torch::Tensor lstm_input = torch::ones({seq_len, batch_size, input_size}, torch::kFloat);
            
            // Apply the module
            auto lstm_output = lstm_module->forward(lstm_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
