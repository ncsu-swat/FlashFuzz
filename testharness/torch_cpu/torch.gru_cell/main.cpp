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
            // If we don't have enough data, create a default hidden state
            if (input.dim() > 0 && input.size(0) > 0) {
                int64_t batch_size = input.size(0);
                int64_t hidden_size = 4; // Small default hidden size
                hx = torch::zeros({batch_size, hidden_size});
            } else {
                // For scalar input or empty batch, create a minimal hidden state
                hx = torch::zeros({1, 4});
            }
        }
        
        // Create weight tensors
        torch::Tensor w_ih, w_hh, b_ih, b_hh;
        
        // Create input-hidden weights
        if (offset < Size) {
            w_ih = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create default weight tensor
            int64_t input_size = input.dim() > 1 ? input.size(1) : 1;
            int64_t hidden_size = hx.dim() > 1 ? hx.size(1) : 1;
            w_ih = torch::randn({3 * hidden_size, input_size});
        }
        
        // Create hidden-hidden weights
        if (offset < Size) {
            w_hh = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create default weight tensor
            int64_t hidden_size = hx.dim() > 1 ? hx.size(1) : 1;
            w_hh = torch::randn({3 * hidden_size, hidden_size});
        }
        
        // Create bias tensors (optional)
        bool use_bias = offset < Size && Data[offset++] % 2 == 0;
        
        if (use_bias) {
            // Create input bias
            if (offset < Size) {
                b_ih = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                int64_t hidden_size = hx.dim() > 1 ? hx.size(1) : 1;
                b_ih = torch::randn({3 * hidden_size});
            }
            
            // Create hidden bias
            if (offset < Size) {
                b_hh = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                int64_t hidden_size = hx.dim() > 1 ? hx.size(1) : 1;
                b_hh = torch::randn({3 * hidden_size});
            }
        }
        
        // Apply GRU cell operation
        torch::Tensor output;
        
        if (use_bias) {
            output = torch::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
        } else {
            output = torch::gru_cell(input, hx, w_ih, w_hh);
        }
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
