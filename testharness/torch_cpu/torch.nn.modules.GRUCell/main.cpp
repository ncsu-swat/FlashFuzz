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
            // If we don't have enough data, create a default hidden state
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
                // Default hidden state for empty input
                hx = torch::zeros({1, 10});
            }
        }
        
        // Extract parameters for GRUCell
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        bool bias = true;
        
        // Determine input_size from input tensor if possible
        if (input.dim() >= 2) {
            input_size = input.size(1);
        } else if (input.dim() == 1) {
            input_size = input.size(0);
        } else {
            // Default input size
            input_size = 10;
        }
        
        // Determine hidden_size from hx tensor if possible
        if (hx.dim() >= 2) {
            hidden_size = hx.size(1);
        } else if (hx.dim() == 1) {
            hidden_size = hx.size(0);
        } else {
            // Default hidden size
            hidden_size = 10;
        }
        
        // Determine bias parameter
        if (offset < Size) {
            bias = Data[offset++] & 1; // Use lowest bit to determine bias
        }
        
        // Create GRUCell
        torch::nn::GRUCell gru_cell(
            torch::nn::GRUCellOptions(input_size, hidden_size).bias(bias)
        );
        
        // Reshape input tensor if needed to match expected dimensions [batch_size, input_size]
        if (input.dim() == 0) {
            // Scalar input - reshape to [1, 1]
            input = input.reshape({1, 1});
        } else if (input.dim() == 1) {
            // 1D input - reshape to [1, input_size]
            input = input.reshape({1, input.size(0)});
        } else if (input.dim() > 2) {
            // Higher dimensional input - flatten to 2D
            int64_t batch_size = input.size(0);
            input = input.reshape({batch_size, -1});
        }
        
        // Reshape hidden state if needed to match expected dimensions [batch_size, hidden_size]
        if (hx.dim() == 0) {
            // Scalar hidden state - reshape to [1, 1]
            hx = hx.reshape({1, 1});
        } else if (hx.dim() == 1) {
            // 1D hidden state - reshape to [1, hidden_size]
            hx = hx.reshape({1, hx.size(0)});
        } else if (hx.dim() > 2) {
            // Higher dimensional hidden state - flatten to 2D
            int64_t batch_size = hx.size(0);
            hx = hx.reshape({batch_size, -1});
        }
        
        // Make sure batch sizes match between input and hidden state
        if (input.size(0) != hx.size(0)) {
            // Adjust hidden state batch size to match input
            int64_t batch_size = input.size(0);
            hx = hx.repeat({batch_size, 1});
            hx = hx.slice(0, 0, batch_size);
        }
        
        // Apply GRUCell operation
        torch::Tensor output = gru_cell(input, hx);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        auto mean = output.mean();
        auto max_val = output.max();
        
        // Prevent compiler from optimizing away the operations
        if (sum.item<float>() == -1.0f && mean.item<float>() == -1.0f && max_val.item<float>() == -1.0f) {
            return 1; // This condition is unlikely to be true
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}