#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Parse input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse hidden state tensor
        torch::Tensor h_0;
        torch::Tensor c_0;
        
        if (offset < Size) {
            h_0 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            h_0 = torch::zeros({1, 10});
        }
        
        if (offset < Size) {
            c_0 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            c_0 = torch::zeros({1, 10});
        }
        
        // Extract parameters for LSTM cell
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        
        if (input.dim() > 0 && input.size(input.dim() - 1) > 0) {
            input_size = input.size(input.dim() - 1);
        } else {
            input_size = 10;
        }
        
        if (h_0.dim() > 0 && h_0.size(h_0.dim() - 1) > 0) {
            hidden_size = h_0.size(h_0.dim() - 1);
        } else {
            hidden_size = 10;
        }
        
        // Ensure input has correct shape for LSTM cell
        if (input.dim() == 0) {
            input = input.reshape({1, input_size});
        } else if (input.dim() == 1) {
            input = input.reshape({1, input.size(0)});
        }
        
        // Ensure hidden states have correct shape
        if (h_0.dim() == 0) {
            h_0 = h_0.reshape({1, hidden_size});
        } else if (h_0.dim() == 1) {
            h_0 = h_0.reshape({1, h_0.size(0)});
        }
        
        if (c_0.dim() == 0) {
            c_0 = c_0.reshape({1, hidden_size});
        } else if (c_0.dim() == 1) {
            c_0 = c_0.reshape({1, c_0.size(0)});
        }
        
        // Try to make batch sizes consistent
        int64_t batch_size = 1;
        if (input.dim() > 1) {
            batch_size = input.size(0);
        }
        
        if (h_0.dim() > 1 && h_0.size(0) != batch_size) {
            h_0 = h_0.repeat({batch_size, 1});
        }
        
        if (c_0.dim() > 1 && c_0.size(0) != batch_size) {
            c_0 = c_0.repeat({batch_size, 1});
        }
        
        // Create weight matrices for LSTM cell
        torch::Tensor w_ih = torch::randn({4 * hidden_size, input_size});
        torch::Tensor w_hh = torch::randn({4 * hidden_size, hidden_size});
        torch::Tensor b_ih = torch::randn({4 * hidden_size});
        torch::Tensor b_hh = torch::randn({4 * hidden_size});
        
        // Apply the LSTM cell using the functional interface
        std::vector<torch::Tensor> hx = {h_0, c_0};
        auto result = torch::lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
        
        // Extract the output hidden state and cell state
        auto h_1 = std::get<0>(result);
        auto c_1 = std::get<1>(result);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}