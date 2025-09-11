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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a quantized linear dynamic module
        int64_t in_features = 0;
        int64_t out_features = 0;
        
        if (input_tensor.dim() >= 2) {
            in_features = input_tensor.size(-1);
            out_features = in_features > 0 ? (in_features % 8) + 1 : 1;
        } else if (input_tensor.dim() == 1) {
            in_features = input_tensor.size(0);
            out_features = in_features > 0 ? (in_features % 8) + 1 : 1;
        } else {
            in_features = 4;
            out_features = 2;
        }
        
        // Try different quantized dynamic modules
        if (offset < Size) {
            uint8_t module_type = Data[offset++] % 4;
            
            switch (module_type) {
                case 0: {
                    // Linear with ReLU
                    torch::nn::Linear linear(in_features, out_features);
                    torch::nn::ReLU relu;
                    
                    // Apply the modules
                    torch::Tensor output = relu(linear(input_tensor));
                    break;
                }
                case 1: {
                    // LSTM
                    int64_t hidden_size = out_features;
                    int64_t num_layers = 1;
                    bool bias = true;
                    bool batch_first = true;
                    bool bidirectional = false;
                    
                    if (offset + 3 < Size) {
                        hidden_size = (Data[offset++] % 8) + 1;
                        num_layers = (Data[offset++] % 3) + 1;
                        bias = Data[offset++] % 2;
                        batch_first = Data[offset++] % 2;
                        bidirectional = Data[offset++] % 2;
                    }
                    
                    torch::nn::LSTM lstm(
                        torch::nn::LSTMOptions(in_features, hidden_size)
                            .num_layers(num_layers)
                            .bias(bias)
                            .batch_first(batch_first)
                            .bidirectional(bidirectional));
                    
                    // Prepare input for LSTM
                    torch::Tensor lstm_input;
                    if (input_tensor.dim() < 2) {
                        // LSTM expects at least 2D input
                        lstm_input = input_tensor.view({1, -1});
                    } else if (input_tensor.dim() == 2 && !batch_first) {
                        // If not batch_first, input should be [seq_len, batch, input_size]
                        lstm_input = input_tensor.unsqueeze(1);
                    } else {
                        lstm_input = input_tensor;
                    }
                    
                    // Apply the module
                    auto lstm_output = lstm(lstm_input);
                    break;
                }
                case 2: {
                    // GRU
                    int64_t hidden_size = out_features;
                    int64_t num_layers = 1;
                    bool bias = true;
                    bool batch_first = true;
                    bool bidirectional = false;
                    
                    if (offset + 3 < Size) {
                        hidden_size = (Data[offset++] % 8) + 1;
                        num_layers = (Data[offset++] % 3) + 1;
                        bias = Data[offset++] % 2;
                        batch_first = Data[offset++] % 2;
                        bidirectional = Data[offset++] % 2;
                    }
                    
                    torch::nn::GRU gru(
                        torch::nn::GRUOptions(in_features, hidden_size)
                            .num_layers(num_layers)
                            .bias(bias)
                            .batch_first(batch_first)
                            .bidirectional(bidirectional));
                    
                    // Prepare input for GRU
                    torch::Tensor gru_input;
                    if (input_tensor.dim() < 2) {
                        // GRU expects at least 2D input
                        gru_input = input_tensor.view({1, -1});
                    } else if (input_tensor.dim() == 2 && !batch_first) {
                        // If not batch_first, input should be [seq_len, batch, input_size]
                        gru_input = input_tensor.unsqueeze(1);
                    } else {
                        gru_input = input_tensor;
                    }
                    
                    // Apply the module
                    auto gru_output = gru(gru_input);
                    break;
                }
                case 3: {
                    // Linear
                    torch::nn::Linear linear(in_features, out_features);
                    
                    // Apply the module
                    torch::Tensor output = linear(input_tensor);
                    break;
                }
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
