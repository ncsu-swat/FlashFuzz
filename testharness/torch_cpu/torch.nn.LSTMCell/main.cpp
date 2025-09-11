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
        
        // Create hidden state tensors (h0, c0)
        torch::Tensor h0, c0;
        
        // If we have more data, create h0 and c0
        if (offset < Size) {
            h0 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create default h0 with same batch size as input but hidden_size=4
            if (input.dim() >= 1) {
                int64_t batch_size = input.size(0);
                h0 = torch::zeros({batch_size, 4}, input.options());
            } else {
                h0 = torch::zeros({1, 4}, input.options());
            }
        }
        
        if (offset < Size) {
            c0 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create default c0 with same shape as h0
            c0 = torch::zeros_like(h0);
        }
        
        // Get input size from the input tensor
        int64_t input_size = 1;
        if (input.dim() >= 2) {
            input_size = input.size(1);
        } else if (input.dim() == 1) {
            input_size = input.size(0);
        }
        
        // Get hidden size from h0
        int64_t hidden_size = 1;
        if (h0.dim() >= 2) {
            hidden_size = h0.size(1);
        } else if (h0.dim() == 1) {
            hidden_size = h0.size(0);
        }
        
        // Create LSTM cell with input_size and hidden_size
        torch::nn::LSTMCellOptions options(input_size, hidden_size);
        
        // Set bias parameter based on a byte from the input if available
        if (offset < Size) {
            bool use_bias = (Data[offset++] % 2 == 0);
            options.bias(use_bias);
        }
        
        torch::nn::LSTMCell lstm_cell(options);
        
        // Apply the LSTM cell
        auto result = lstm_cell(input, std::make_tuple(h0, c0));
        
        // Extract the output tensors
        torch::Tensor h1 = std::get<0>(result);
        torch::Tensor c1 = std::get<1>(result);
        
        // Perform some operations on the output to ensure it's used
        auto sum_h = torch::sum(h1);
        auto sum_c = torch::sum(c1);
        
        // Prevent the compiler from optimizing away the computation
        if (sum_h.item<float>() == -12345.6789f && sum_c.item<float>() == -12345.6789f) {
            return 1; // This condition is extremely unlikely
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
