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
            // If we don't have enough data for a second tensor, create a compatible one
            if (input.dim() > 0 && input.size(0) > 0) {
                int64_t batch_size = input.size(0);
                int64_t hidden_size = 10; // Default hidden size
                
                // Extract hidden size from remaining bytes if available
                if (offset + 1 < Size) {
                    hidden_size = static_cast<int64_t>(Data[offset++]) % 100 + 1;
                }
                
                hx = torch::zeros({batch_size, hidden_size});
            } else {
                // Default hidden state if input is not suitable
                hx = torch::zeros({1, 10});
            }
        }
        
        // Determine input size and hidden size
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        
        if (input.dim() >= 2) {
            input_size = input.size(1);
        } else if (input.dim() == 1 && input.size(0) > 0) {
            input_size = input.size(0);
            input = input.unsqueeze(0); // Add batch dimension
        } else {
            input_size = 10; // Default
            input = torch::zeros({1, input_size});
        }
        
        if (hx.dim() >= 2) {
            hidden_size = hx.size(1);
        } else if (hx.dim() == 1 && hx.size(0) > 0) {
            hidden_size = hx.size(0);
            hx = hx.unsqueeze(0); // Add batch dimension
        } else {
            hidden_size = 10; // Default
            hx = torch::zeros({1, hidden_size});
        }
        
        // Make sure input and hidden state have compatible batch size
        if (input.dim() > 0 && hx.dim() > 0 && input.size(0) != hx.size(0)) {
            int64_t batch_size = std::min(input.size(0), hx.size(0));
            input = input.slice(0, 0, batch_size);
            hx = hx.slice(0, 0, batch_size);
        }
        
        // Ensure tensors have float dtype for quantization
        input = input.to(torch::kFloat);
        hx = hx.to(torch::kFloat);
        
        // Create weight tensors for GRU cell
        torch::Tensor w_ih = torch::randn({3 * hidden_size, input_size});
        torch::Tensor w_hh = torch::randn({3 * hidden_size, hidden_size});
        torch::Tensor b_ih = torch::randn({3 * hidden_size});
        torch::Tensor b_hh = torch::randn({3 * hidden_size});
        
        // Apply the GRU cell operation using functional interface
        torch::Tensor output = torch::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
        
        // Try without bias if we have more data
        if (offset + 1 < Size) {
            bool use_bias = Data[offset++] % 2 == 0;
            if (use_bias) {
                output = torch::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
            } else {
                output = torch::gru_cell(input, hx, w_ih, w_hh);
            }
        }
        
        // Try with different weight initialization if we have more data
        if (offset + 1 < Size) {
            double scale = static_cast<double>(Data[offset++]) / 255.0 + 0.1;
            w_ih = torch::randn({3 * hidden_size, input_size}) * scale;
            w_hh = torch::randn({3 * hidden_size, hidden_size}) * scale;
            output = torch::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
