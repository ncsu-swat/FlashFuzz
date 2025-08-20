#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
            if (input.dim() > 0 && input.size(0) > 0) {
                auto batch_size = input.size(0);
                auto hidden_size = input.size(input.dim() - 1);
                hx = torch::zeros({batch_size, hidden_size}, input.options());
            } else {
                // Default hidden state if input is empty or scalar
                hx = torch::zeros({1, 1}, input.options());
            }
        }
        
        // Create weight tensors
        torch::Tensor w_ih, w_hh, b_ih, b_hh;
        
        // Try to create weight tensors from the input data
        if (offset < Size) {
            w_ih = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create compatible weight tensor
            int64_t input_size = (input.dim() > 0) ? input.size(input.dim() - 1) : 1;
            int64_t hidden_size = (hx.dim() > 0) ? hx.size(hx.dim() - 1) : 1;
            w_ih = torch::randn({hidden_size, input_size}, input.options());
        }
        
        if (offset < Size) {
            w_hh = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create compatible weight tensor
            int64_t hidden_size = (hx.dim() > 0) ? hx.size(hx.dim() - 1) : 1;
            w_hh = torch::randn({hidden_size, hidden_size}, input.options());
        }
        
        // Create bias tensors (optional)
        bool use_bias = (offset < Size && Data[offset++] % 2 == 0);
        
        if (use_bias) {
            if (offset < Size) {
                b_ih = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                int64_t hidden_size = (hx.dim() > 0) ? hx.size(hx.dim() - 1) : 1;
                b_ih = torch::randn({hidden_size}, input.options());
            }
            
            if (offset < Size) {
                b_hh = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                int64_t hidden_size = (hx.dim() > 0) ? hx.size(hx.dim() - 1) : 1;
                b_hh = torch::randn({hidden_size}, input.options());
            }
        }
        
        // Apply the RNN ReLU cell operation
        torch::Tensor output;
        
        if (use_bias) {
            output = torch::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
        } else {
            output = torch::rnn_relu_cell(input, hx, w_ih, w_hh);
        }
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        if (sum.item<float>() == -1.0f) {
            // This is just to use the result and avoid compiler optimizations
            // The condition is unlikely to be true
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}