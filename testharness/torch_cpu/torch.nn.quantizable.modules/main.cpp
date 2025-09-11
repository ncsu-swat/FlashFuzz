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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create quantizable modules
        torch::nn::Linear linear = torch::nn::Linear(input.size(-1), input.size(-1));
        torch::nn::Conv1d conv1d = torch::nn::Conv1d(torch::nn::Conv1dOptions(input.size(1), input.size(1), 3).padding(1));
        torch::nn::Conv2d conv2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(input.size(1), input.size(1), 3).padding(1));
        torch::nn::Conv3d conv3d = torch::nn::Conv3d(torch::nn::Conv3dOptions(input.size(1), input.size(1), 3).padding(1));
        
        // Create quantizable RNN modules
        int64_t input_size = input.size(-1);
        int64_t hidden_size = input_size > 0 ? input_size : 1;
        torch::nn::LSTM lstm = torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size));
        torch::nn::GRU gru = torch::nn::GRU(torch::nn::GRUOptions(input_size, hidden_size));
        torch::nn::RNN rnn = torch::nn::RNN(torch::nn::RNNOptions(input_size, hidden_size));
        
        // Apply operations based on tensor rank
        if (input.dim() >= 2) {
            // Linear layer
            auto linear_output = linear->forward(input);
            
            // LSTM, GRU, RNN
            if (input.dim() == 3) {
                auto lstm_output = lstm->forward(input);
                auto gru_output = gru->forward(input);
                auto rnn_output = rnn->forward(input);
            }
            
            // Convolutional layers
            if (input.dim() == 3) {
                auto conv1d_output = conv1d->forward(input);
            } else if (input.dim() == 4) {
                auto conv2d_output = conv2d->forward(input);
            } else if (input.dim() == 5) {
                auto conv3d_output = conv3d->forward(input);
            }
        }
        
        // Test quantization-related functionality
        if (offset + 1 < Size) {
            uint8_t qconfig_selector = Data[offset++];
            
            // Create a quantization config
            torch::nn::Sequential model = torch::nn::Sequential(
                linear,
                torch::nn::ReLU()
            );
            
            // Prepare model for quantization
            model->train();
            
            // Fuzz different quantization configurations
            if (qconfig_selector % 3 == 0) {
                // Test with dummy input for quantization
                if (input.dim() >= 2) {
                    model->forward(input);
                }
            } else if (qconfig_selector % 3 == 1) {
                // Test with different dtype
                if (input.dim() >= 2) {
                    auto float_input = input.to(torch::kFloat);
                    model->forward(float_input);
                }
            } else {
                // Test with different batch size
                if (input.dim() >= 2) {
                    auto reshaped_input = input;
                    if (input.size(0) > 1) {
                        reshaped_input = input.slice(0, 0, input.size(0) / 2);
                    }
                    model->forward(reshaped_input);
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
