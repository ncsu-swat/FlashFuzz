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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create hidden state tensor
        torch::Tensor hx;
        if (offset < Size) {
            hx = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for hx, create a compatible one
            if (input.dim() > 0 && input.size(0) > 0) {
                hx = torch::zeros({input.size(0), input.size(-1)});
            } else {
                hx = torch::zeros({1, 10});
            }
        }
        
        // Extract parameters for GRUCell
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        
        // Determine input_size and hidden_size from input tensor if possible
        if (input.dim() >= 2) {
            input_size = input.size(-1);
        } else if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&input_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            input_size = std::abs(input_size) % 100 + 1; // Ensure positive and reasonable size
        } else {
            input_size = 10; // Default value
        }
        
        if (hx.dim() >= 2) {
            hidden_size = hx.size(-1);
        } else if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&hidden_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            hidden_size = std::abs(hidden_size) % 100 + 1; // Ensure positive and reasonable size
        } else {
            hidden_size = 20; // Default value
        }
        
        // Create weight tensors for GRU cell
        torch::Tensor w_ih = torch::randn({3 * hidden_size, input_size});
        torch::Tensor w_hh = torch::randn({3 * hidden_size, hidden_size});
        
        // Create bias tensors if we have more data
        torch::Tensor b_ih, b_hh;
        bool use_bias = false;
        if (offset < Size) {
            use_bias = Data[offset++] % 2 == 0;
            if (use_bias) {
                b_ih = torch::randn({3 * hidden_size});
                b_hh = torch::randn({3 * hidden_size});
            }
        }
        
        // Ensure input and hx have compatible shapes for GRUCell
        if (input.dim() == 0) {
            input = input.reshape({1, input_size});
        } else if (input.dim() == 1) {
            input = input.reshape({1, input.size(0)});
            if (input.size(1) != input_size) {
                input = input.narrow(1, 0, std::min(input.size(1), input_size));
                if (input.size(1) < input_size) {
                    auto padding = torch::zeros({input.size(0), input_size - input.size(1)});
                    input = torch::cat({input, padding}, 1);
                }
            }
        }
        
        if (hx.dim() == 0) {
            hx = hx.reshape({1, hidden_size});
        } else if (hx.dim() == 1) {
            hx = hx.reshape({1, hx.size(0)});
            if (hx.size(1) != hidden_size) {
                hx = hx.narrow(1, 0, std::min(hx.size(1), hidden_size));
                if (hx.size(1) < hidden_size) {
                    auto padding = torch::zeros({hx.size(0), hidden_size - hx.size(1)});
                    hx = torch::cat({hx, padding}, 1);
                }
            }
        }
        
        // Ensure batch sizes match
        if (input.dim() > 1 && hx.dim() > 1 && input.size(0) != hx.size(0)) {
            int64_t batch_size = std::min(input.size(0), hx.size(0));
            input = input.narrow(0, 0, batch_size);
            hx = hx.narrow(0, 0, batch_size);
        }
        
        // Apply the GRUCell operation
        torch::Tensor output;
        if (use_bias) {
            output = torch::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
        } else {
            output = torch::gru_cell(input, hx, w_ih, w_hh);
        }
        
        // Use the output to prevent optimization from removing the computation
        if (output.numel() > 0) {
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