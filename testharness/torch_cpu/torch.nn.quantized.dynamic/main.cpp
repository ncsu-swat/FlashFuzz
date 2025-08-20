#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get parameters for dynamic quantization
        uint8_t dtype_selector = (offset < Size) ? Data[offset++] : 0;
        uint8_t qscheme_selector = (offset < Size) ? Data[offset++] : 0;
        
        // Select dtype for quantization
        torch::ScalarType dtype;
        if (dtype_selector % 2 == 0) {
            dtype = torch::kQInt8;
        } else {
            dtype = torch::kQUInt8;
        }
        
        // Select quantization scheme
        torch::QScheme qscheme;
        switch (qscheme_selector % 3) {
            case 0:
                qscheme = torch::kPerTensorAffine;
                break;
            case 1:
                qscheme = torch::kPerChannelAffine;
                break;
            default:
                qscheme = torch::kPerTensorSymmetric;
                break;
        }
        
        // Create a linear layer for dynamic quantization
        int64_t in_features = 0;
        int64_t out_features = 0;
        
        if (input_tensor.dim() > 0) {
            in_features = input_tensor.size(-1);
        } else {
            in_features = 4; // Default value for scalar input
        }
        
        // Get out_features from the input data if available
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_features = std::abs(out_features) % 32 + 1; // Ensure positive and reasonable size
        } else {
            out_features = 4; // Default value
        }
        
        // Get bias flag from input data
        bool with_bias = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        
        // Create regular linear module and apply dynamic quantization
        auto linear_options = torch::nn::LinearOptions(in_features, out_features).bias(with_bias);
        torch::nn::Linear linear_module(linear_options);
        
        // Try to apply the linear operation
        torch::Tensor output;
        
        // Reshape input if needed to match expected dimensions for linear layer
        if (input_tensor.dim() == 0) {
            // For scalar input, reshape to [1, in_features]
            input_tensor = input_tensor.reshape({1, in_features});
        } else if (input_tensor.dim() == 1) {
            // For 1D input, reshape to [1, in_features]
            if (input_tensor.size(0) != in_features) {
                input_tensor = input_tensor.reshape({1, in_features});
            }
        }
        
        // Apply the linear operation
        output = linear_module->forward(input_tensor);
        
        // Apply dynamic quantization to the output
        torch::Tensor quantized_output = torch::quantize_per_tensor(
            output, 
            1.0, // scale
            0,   // zero_point
            dtype
        );
        
        // Test other RNN modules if there's more data
        if (offset < Size) {
            uint8_t module_selector = Data[offset++];
            
            // Create LSTM if selected
            if (module_selector % 3 == 0) {
                int64_t input_size = in_features;
                int64_t hidden_size = out_features;
                int num_layers = (offset < Size) ? (Data[offset++] % 3) + 1 : 1;
                bool bias = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
                
                auto lstm_options = torch::nn::LSTMOptions(input_size, hidden_size)
                                        .num_layers(num_layers)
                                        .bias(bias);
                
                torch::nn::LSTM lstm(lstm_options);
                
                // Prepare input for LSTM
                torch::Tensor lstm_input;
                if (input_tensor.dim() < 3) {
                    // LSTM expects [seq_len, batch, input_size]
                    int64_t seq_len = 1;
                    int64_t batch = 1;
                    
                    if (input_tensor.dim() == 2) {
                        seq_len = input_tensor.size(0);
                        lstm_input = input_tensor.reshape({seq_len, batch, input_size});
                    } else {
                        lstm_input = input_tensor.reshape({seq_len, batch, input_size});
                    }
                } else {
                    lstm_input = input_tensor;
                }
                
                // Forward pass through LSTM
                auto lstm_output = lstm->forward(lstm_input);
            }
            
            // Create GRU if selected
            else if (module_selector % 3 == 1) {
                int64_t input_size = in_features;
                int64_t hidden_size = out_features;
                int num_layers = (offset < Size) ? (Data[offset++] % 3) + 1 : 1;
                bool bias = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
                
                auto gru_options = torch::nn::GRUOptions(input_size, hidden_size)
                                      .num_layers(num_layers)
                                      .bias(bias);
                
                torch::nn::GRU gru(gru_options);
                
                // Prepare input for GRU
                torch::Tensor gru_input;
                if (input_tensor.dim() < 3) {
                    // GRU expects [seq_len, batch, input_size]
                    int64_t seq_len = 1;
                    int64_t batch = 1;
                    
                    if (input_tensor.dim() == 2) {
                        seq_len = input_tensor.size(0);
                        gru_input = input_tensor.reshape({seq_len, batch, input_size});
                    } else {
                        gru_input = input_tensor.reshape({seq_len, batch, input_size});
                    }
                } else {
                    gru_input = input_tensor;
                }
                
                // Forward pass through GRU
                auto gru_output = gru->forward(gru_input);
            }
            
            // Create RNN if selected
            else {
                int64_t input_size = in_features;
                int64_t hidden_size = out_features;
                int num_layers = (offset < Size) ? (Data[offset++] % 3) + 1 : 1;
                bool bias = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
                
                auto rnn_options = torch::nn::RNNOptions(input_size, hidden_size)
                                      .num_layers(num_layers)
                                      .bias(bias);
                
                torch::nn::RNN rnn(rnn_options);
                
                // Prepare input for RNN
                torch::Tensor rnn_input;
                if (input_tensor.dim() < 3) {
                    // RNN expects [seq_len, batch, input_size]
                    int64_t seq_len = 1;
                    int64_t batch = 1;
                    
                    if (input_tensor.dim() == 2) {
                        seq_len = input_tensor.size(0);
                        rnn_input = input_tensor.reshape({seq_len, batch, input_size});
                    } else {
                        rnn_input = input_tensor.reshape({seq_len, batch, input_size});
                    }
                } else {
                    rnn_input = input_tensor;
                }
                
                // Forward pass through RNN
                auto rnn_output = rnn->forward(rnn_input);
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