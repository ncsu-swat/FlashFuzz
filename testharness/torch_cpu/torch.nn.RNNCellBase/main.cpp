#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;
        
        // Extract configuration parameters from fuzzer data
        uint8_t cell_type = Data[offset++] % 3;  // 0: RNN, 1: LSTM, 2: GRU
        uint8_t batch_size_raw = Data[offset++];
        uint8_t input_size_raw = Data[offset++];
        uint8_t hidden_size_raw = Data[offset++];
        bool bias = Data[offset++] % 2 == 0;
        uint8_t nonlinearity_choice = Data[offset++] % 2;  // 0: tanh, 1: relu
        
        // Constrain sizes to reasonable ranges
        int64_t batch_size = 1 + (batch_size_raw % 16);    // 1-16
        int64_t input_size = 1 + (input_size_raw % 32);    // 1-32
        int64_t hidden_size = 1 + (hidden_size_raw % 32);  // 1-32
        
        // Create input tensor [batch_size, input_size]
        torch::Tensor input = torch::randn({batch_size, input_size}, torch::kFloat32);
        
        // Optionally modify input values using fuzzer data
        if (offset + sizeof(float) <= Size) {
            float scale = *reinterpret_cast<const float*>(Data + offset);
            offset += sizeof(float);
            if (std::isfinite(scale) && std::abs(scale) < 100.0f) {
                input = input * scale;
            }
        }
        
        try {
            if (cell_type == 0) {
                // Test RNNCell
                auto options = torch::nn::RNNCellOptions(input_size, hidden_size)
                    .bias(bias);
                
                // Set nonlinearity based on choice - must be done separately due to different enum types
                if (nonlinearity_choice == 0) {
                    options.nonlinearity(torch::kTanh);
                } else {
                    options.nonlinearity(torch::kReLU);
                }
                
                torch::nn::RNNCell cell(options);
                
                // Create hidden state [batch_size, hidden_size]
                torch::Tensor hx = torch::randn({batch_size, hidden_size}, torch::kFloat32);
                
                // Forward pass
                torch::Tensor output = cell->forward(input, hx);
                
                // Also test forward without initial hidden state
                torch::Tensor output2 = cell->forward(input);
                
                // Verify output shape
                assert(output.size(0) == batch_size);
                assert(output.size(1) == hidden_size);
            }
            else if (cell_type == 1) {
                // Test LSTMCell
                auto options = torch::nn::LSTMCellOptions(input_size, hidden_size)
                    .bias(bias);
                
                torch::nn::LSTMCell cell(options);
                
                // Create hidden state tuple (hx, cx)
                torch::Tensor hx = torch::randn({batch_size, hidden_size}, torch::kFloat32);
                torch::Tensor cx = torch::randn({batch_size, hidden_size}, torch::kFloat32);
                
                // Forward pass with hidden state
                auto output_tuple = cell->forward(input, std::make_tuple(hx, cx));
                
                // Also test forward without initial hidden state
                auto output_tuple2 = cell->forward(input);
                
                // Verify output shapes
                assert(std::get<0>(output_tuple).size(0) == batch_size);
                assert(std::get<0>(output_tuple).size(1) == hidden_size);
                assert(std::get<1>(output_tuple).size(0) == batch_size);
                assert(std::get<1>(output_tuple).size(1) == hidden_size);
            }
            else {
                // Test GRUCell
                auto options = torch::nn::GRUCellOptions(input_size, hidden_size)
                    .bias(bias);
                
                torch::nn::GRUCell cell(options);
                
                // Create hidden state [batch_size, hidden_size]
                torch::Tensor hx = torch::randn({batch_size, hidden_size}, torch::kFloat32);
                
                // Forward pass
                torch::Tensor output = cell->forward(input, hx);
                
                // Also test forward without initial hidden state
                torch::Tensor output2 = cell->forward(input);
                
                // Verify output shape
                assert(output.size(0) == batch_size);
                assert(output.size(1) == hidden_size);
            }
        }
        catch (const c10::Error&) {
            // Expected errors for invalid configurations
        }
        catch (const std::runtime_error&) {
            // Expected runtime errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}