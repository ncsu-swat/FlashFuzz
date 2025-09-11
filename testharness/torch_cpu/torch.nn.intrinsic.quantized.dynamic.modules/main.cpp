#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a byte to determine which module to test
        uint8_t module_selector = 0;
        if (offset < Size) {
            module_selector = Data[offset++];
        }
        
        // Get a byte for configuration options
        uint8_t config_byte = 0;
        if (offset < Size) {
            config_byte = Data[offset++];
        }
        
        // Create quantized dynamic modules and test them
        switch (module_selector % 3) {
            case 0: {
                // Test Linear with quantized dynamic approach
                int64_t in_features = 0;
                int64_t out_features = 0;
                
                if (input_tensor.dim() > 0) {
                    in_features = input_tensor.size(-1);
                } else {
                    in_features = 4;
                }
                
                out_features = (config_byte % 8) + 1;
                
                torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features));
                
                // Apply the module
                torch::Tensor output = linear(input_tensor);
                
                // Apply ReLU activation
                torch::Tensor relu_output = torch::relu(output);
                break;
            }
            
            case 1: {
                // Test Linear
                int64_t in_features = 0;
                int64_t out_features = 0;
                
                if (input_tensor.dim() > 0) {
                    in_features = input_tensor.size(-1);
                } else {
                    in_features = 4;
                }
                
                out_features = (config_byte % 8) + 1;
                
                torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features));
                
                // Apply the module
                torch::Tensor output = linear(input_tensor);
                break;
            }
            
            case 2: {
                // Test LSTM
                int64_t input_size = 0;
                int64_t hidden_size = 0;
                int64_t num_layers = (config_byte % 3) + 1;
                bool bias = (config_byte & 0x10) != 0;
                bool batch_first = (config_byte & 0x20) != 0;
                bool bidirectional = (config_byte & 0x40) != 0;
                double dropout = 0.0;
                
                if (input_tensor.dim() > 0) {
                    input_size = input_tensor.size(-1);
                } else {
                    input_size = 4;
                }
                
                hidden_size = (config_byte % 4) + 1;
                
                torch::nn::LSTM lstm(torch::nn::LSTMOptions(input_size, hidden_size)
                    .num_layers(num_layers)
                    .bias(bias)
                    .batch_first(batch_first)
                    .bidirectional(bidirectional)
                    .dropout(dropout));
                
                // Reshape input tensor if needed for LSTM
                torch::Tensor reshaped_input = input_tensor;
                if (input_tensor.dim() < 2) {
                    // LSTM expects at least 2D tensor (seq_len, input_size) or (batch, seq_len, input_size)
                    if (batch_first) {
                        reshaped_input = input_tensor.reshape({1, 1, input_size});
                    } else {
                        reshaped_input = input_tensor.reshape({1, 1, input_size});
                    }
                } else if (input_tensor.dim() == 2) {
                    if (batch_first) {
                        reshaped_input = input_tensor.unsqueeze(1); // Add seq_len dimension
                    } else {
                        reshaped_input = input_tensor.unsqueeze(0); // Add batch dimension
                    }
                }
                
                // Apply the module
                auto output = lstm(reshaped_input);
                break;
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
