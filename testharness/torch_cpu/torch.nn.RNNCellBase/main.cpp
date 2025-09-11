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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create hidden state tensor
        torch::Tensor hx;
        if (offset < Size) {
            hx = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a compatible hidden state
            if (input.dim() > 0) {
                auto batch_size = input.size(0);
                auto hidden_size = 10 + (Data[0] % 20); // Random hidden size
                hx = torch::zeros({batch_size, hidden_size});
            } else {
                // Default hidden state for scalar input
                hx = torch::zeros({1, 10});
            }
        }
        
        // Get parameters for RNNCellBase
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        
        if (input.dim() > 0 && input.size(0) > 0) {
            // Use the last dimension of input as input_size
            if (input.dim() > 1) {
                input_size = input.size(input.dim() - 1);
            } else {
                input_size = 1;
            }
        } else {
            // Default input size
            input_size = 10;
        }
        
        if (hx.dim() > 0 && hx.size(0) > 0) {
            // Use the last dimension of hx as hidden_size
            if (hx.dim() > 1) {
                hidden_size = hx.size(hx.dim() - 1);
            } else {
                hidden_size = 1;
            }
        } else {
            // Default hidden size
            hidden_size = 10;
        }
        
        // Create bias option
        bool bias = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        
        // Create nonlinearity option (for RNNCell)
        std::string nonlinearity = (offset < Size && Data[offset++] % 2 == 0) ? "tanh" : "relu";
        
        // Create RNNCellBase options
        auto options = torch::nn::RNNCellOptions(input_size, hidden_size)
                           .bias(bias);
        
        // Create RNNCell (a concrete implementation of RNNCellBase)
        torch::nn::RNNCell model(options);
        
        // Reshape input if needed to match expected dimensions [batch_size, input_size]
        if (input.dim() == 0) {
            input = input.reshape({1, 1});
        } else if (input.dim() == 1) {
            input = input.reshape({1, input.size(0)});
        }
        
        // Reshape hidden state if needed to match expected dimensions [batch_size, hidden_size]
        if (hx.dim() == 0) {
            hx = hx.reshape({1, 1});
        } else if (hx.dim() == 1) {
            hx = hx.reshape({1, hx.size(0)});
        }
        
        // Make sure input and hidden state have compatible batch sizes
        if (input.size(0) != hx.size(0)) {
            // Adjust hidden state batch size to match input
            auto new_hx = torch::zeros({input.size(0), hx.size(1)}, hx.options());
            for (int64_t i = 0; i < std::min(input.size(0), hx.size(0)); i++) {
                new_hx[i] = hx[i % hx.size(0)];
            }
            hx = new_hx;
        }
        
        // Make sure input has the right input_size in the last dimension
        if (input.size(1) != input_size) {
            auto new_input = torch::zeros({input.size(0), input_size}, input.options());
            for (int64_t i = 0; i < input.size(0); i++) {
                for (int64_t j = 0; j < std::min(input.size(1), input_size); j++) {
                    new_input[i][j] = input[i][j % input.size(1)];
                }
            }
            input = new_input;
        }
        
        // Make sure hidden state has the right hidden_size in the last dimension
        if (hx.size(1) != hidden_size) {
            auto new_hx = torch::zeros({hx.size(0), hidden_size}, hx.options());
            for (int64_t i = 0; i < hx.size(0); i++) {
                for (int64_t j = 0; j < std::min(hx.size(1), hidden_size); j++) {
                    new_hx[i][j] = hx[i][j % hx.size(1)];
                }
            }
            hx = new_hx;
        }
        
        // Convert tensors to the same dtype if needed
        if (input.dtype() != hx.dtype()) {
            if (torch::can_cast(input.scalar_type(), hx.scalar_type())) {
                input = input.to(hx.dtype());
            } else if (torch::can_cast(hx.scalar_type(), input.scalar_type())) {
                hx = hx.to(input.dtype());
            } else {
                // If neither can be cast to the other, convert both to float
                input = input.to(torch::kFloat);
                hx = hx.to(torch::kFloat);
            }
        }
        
        // Forward pass
        torch::Tensor output = model(input, hx);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
