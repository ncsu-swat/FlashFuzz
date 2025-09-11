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
            if (input.dim() > 0 && input.size(0) > 0) {
                int64_t batch_size = input.size(0);
                int64_t hidden_size = 10; // Default hidden size
                
                // Try to extract hidden size from remaining data if available
                if (offset + 1 < Size) {
                    hidden_size = (Data[offset] % 20) + 1; // 1 to 20
                    offset++;
                }
                
                hx = torch::zeros({batch_size, hidden_size});
            } else {
                // Create a default hidden state
                hx = torch::zeros({1, 10});
            }
        }
        
        // Extract parameters for GRUCell
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        
        // Determine input_size from input tensor
        if (input.dim() >= 2) {
            input_size = input.size(1);
        } else if (input.dim() == 1) {
            input_size = input.size(0);
        } else {
            input_size = 1;
        }
        
        // Determine hidden_size from hx tensor
        if (hx.dim() >= 2) {
            hidden_size = hx.size(1);
        } else if (hx.dim() == 1) {
            hidden_size = hx.size(0);
        } else {
            hidden_size = 1;
        }
        
        // Create GRUCell
        torch::nn::GRUCellOptions options(input_size, hidden_size);
        
        // Set bias option if we have more data
        if (offset < Size) {
            bool use_bias = (Data[offset++] % 2) == 0;
            options.bias(use_bias);
        }
        
        auto gru_cell = torch::nn::GRUCell(options);
        
        // Forward pass
        torch::Tensor output = gru_cell(input, hx);
        
        // Ensure we use the result to prevent optimization
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
