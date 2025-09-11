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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor (hidden state)
        torch::Tensor hidden;
        if (offset < Size) {
            hidden = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create input tensor (input)
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Extract parameters for RNNCell
        int64_t input_size = 1;
        int64_t hidden_size = 1;
        
        // If we have more data, use it to set input_size and hidden_size
        if (offset + 2 <= Size) {
            input_size = static_cast<int64_t>(Data[offset++]) + 1; // Ensure positive
            hidden_size = static_cast<int64_t>(Data[offset++]) + 1; // Ensure positive
        }
        
        // Get nonlinearity type
        bool use_tanh = true;
        if (offset < Size) {
            use_tanh = (Data[offset++] % 2 == 0);
        }
        
        // Get bias flag
        bool bias = true;
        if (offset < Size) {
            bias = (Data[offset++] % 2 == 0);
        }
        
        // Create RNNCell
        torch::nn::RNNCellOptions options(input_size, hidden_size);
        options.nonlinearity(use_tanh ? torch::nn::RNNCellOptions::Nonlinearity::Tanh : torch::nn::RNNCellOptions::Nonlinearity::ReLU);
        options.bias(bias);
        
        torch::nn::RNNCell rnn_cell(options);
        
        // Reshape input and hidden if needed to match expected dimensions
        // RNNCell expects input of shape (batch_size, input_size)
        // and hidden of shape (batch_size, hidden_size)
        
        // Try to reshape input to match expected dimensions
        if (input.dim() != 2 || input.size(1) != input_size) {
            // Get batch size from the first dimension if possible
            int64_t batch_size = 1;
            if (input.dim() > 0 && input.size(0) > 0) {
                batch_size = input.size(0);
            }
            
            // Reshape or create new tensor
            try {
                input = input.reshape({batch_size, input_size});
            } catch (...) {
                // If reshape fails, create a new tensor
                input = torch::ones({batch_size, input_size});
            }
        }
        
        // Try to reshape hidden to match expected dimensions
        if (hidden.dim() != 2 || hidden.size(1) != hidden_size) {
            // Get batch size from input
            int64_t batch_size = input.size(0);
            
            // Reshape or create new tensor
            try {
                hidden = hidden.reshape({batch_size, hidden_size});
            } catch (...) {
                // If reshape fails, create a new tensor
                hidden = torch::zeros({batch_size, hidden_size});
            }
        }
        
        // Ensure input and hidden have the same batch size
        if (input.size(0) != hidden.size(0)) {
            int64_t batch_size = std::min(input.size(0), hidden.size(0));
            input = input.slice(0, 0, batch_size);
            hidden = hidden.slice(0, 0, batch_size);
        }
        
        // Apply RNNCell
        torch::Tensor output = rnn_cell(input, hidden);
        
        // Verify output shape
        if (output.dim() != 2 || output.size(1) != hidden_size) {
            throw std::runtime_error("Unexpected output shape");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
