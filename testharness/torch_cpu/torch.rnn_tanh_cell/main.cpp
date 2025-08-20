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
                hx = torch::zeros({batch_size, hidden_size});
            } else {
                // Default hidden state if input tensor is empty or scalar
                hx = torch::zeros({1, 1});
            }
        }
        
        // Create weight tensors
        torch::Tensor w_ih, w_hh, b_ih, b_hh;
        
        // Try to create weight tensors if we have enough data
        if (offset < Size) {
            w_ih = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create compatible weight tensor
            int64_t input_size = input.dim() > 0 ? input.size(input.dim() - 1) : 1;
            int64_t hidden_size = hx.dim() > 0 ? hx.size(hx.dim() - 1) : 1;
            w_ih = torch::randn({hidden_size, input_size});
        }
        
        if (offset < Size) {
            w_hh = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create compatible weight tensor
            int64_t hidden_size = hx.dim() > 0 ? hx.size(hx.dim() - 1) : 1;
            w_hh = torch::randn({hidden_size, hidden_size});
        }
        
        // Create bias tensors (optional)
        bool use_bias = true;
        if (offset < Size) {
            // Use a byte to determine if we should use bias
            use_bias = Data[offset++] % 2 == 0;
        }
        
        if (use_bias) {
            if (offset < Size) {
                b_ih = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                int64_t hidden_size = hx.dim() > 0 ? hx.size(hx.dim() - 1) : 1;
                b_ih = torch::randn({hidden_size});
            }
            
            if (offset < Size) {
                b_hh = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                int64_t hidden_size = hx.dim() > 0 ? hx.size(hx.dim() - 1) : 1;
                b_hh = torch::randn({hidden_size});
            }
        }
        
        // Apply the RNN tanh cell operation
        torch::Tensor output;
        
        if (use_bias) {
            output = torch::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
        } else {
            output = torch::rnn_tanh_cell(input, hx, w_ih, w_hh);
        }
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        if (sum.item<float>() == 0.0f) {
            // Just to use the result and avoid compiler optimizations
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}