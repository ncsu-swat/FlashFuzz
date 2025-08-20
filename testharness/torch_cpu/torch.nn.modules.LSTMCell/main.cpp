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
        
        // Create hidden state tensors (h0, c0)
        torch::Tensor h0, c0;
        
        if (offset < Size) {
            h0 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a default h0
            if (input.dim() > 0 && input.size(0) > 0) {
                h0 = torch::zeros({input.size(0), 10});
            } else {
                h0 = torch::zeros({1, 10});
            }
        }
        
        if (offset < Size) {
            c0 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a default c0
            if (h0.dim() > 0 && h0.size(0) > 0 && h0.size(1) > 0) {
                c0 = torch::zeros({h0.size(0), h0.size(1)});
            } else {
                c0 = torch::zeros({1, 10});
            }
        }
        
        // Extract parameters for LSTM cell
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        bool bias = true;
        
        if (input.dim() > 0) {
            if (input.dim() > 1) {
                input_size = input.size(1);
            } else {
                input_size = 1;
            }
        } else {
            input_size = 1;
        }
        
        if (h0.dim() > 0 && h0.dim() > 1) {
            hidden_size = h0.size(1);
        } else {
            hidden_size = 10;
        }
        
        // Use a byte from the input to determine whether to use bias
        if (offset < Size) {
            bias = (Data[offset++] % 2 == 0);
        }
        
        // Create LSTM cell
        torch::nn::LSTMCellOptions options(input_size, hidden_size);
        options.bias(bias);
        
        torch::nn::LSTMCell lstm_cell(options);
        
        // Try to make input and hidden states compatible
        if (input.dim() == 0) {
            input = input.unsqueeze(0).unsqueeze(0);
            if (input_size > 1) {
                input = input.expand({1, input_size});
            }
        } else if (input.dim() == 1) {
            input = input.unsqueeze(0);
            if (input.size(1) != input_size && input_size > 0) {
                input = input.slice(1, 0, std::min(input.size(1), input_size));
                if (input.size(1) < input_size) {
                    auto padding = torch::zeros({input.size(0), input_size - input.size(1)}, input.options());
                    input = torch::cat({input, padding}, 1);
                }
            }
        }
        
        // Ensure h0 and c0 have compatible shapes
        if (h0.dim() == 0) {
            h0 = h0.unsqueeze(0).unsqueeze(0);
            if (hidden_size > 1) {
                h0 = h0.expand({1, hidden_size});
            }
        } else if (h0.dim() == 1) {
            h0 = h0.unsqueeze(0);
            if (h0.size(1) != hidden_size && hidden_size > 0) {
                h0 = h0.slice(1, 0, std::min(h0.size(1), hidden_size));
                if (h0.size(1) < hidden_size) {
                    auto padding = torch::zeros({h0.size(0), hidden_size - h0.size(1)}, h0.options());
                    h0 = torch::cat({h0, padding}, 1);
                }
            }
        }
        
        if (c0.dim() == 0) {
            c0 = c0.unsqueeze(0).unsqueeze(0);
            if (hidden_size > 1) {
                c0 = c0.expand({1, hidden_size});
            }
        } else if (c0.dim() == 1) {
            c0 = c0.unsqueeze(0);
            if (c0.size(1) != hidden_size && hidden_size > 0) {
                c0 = c0.slice(1, 0, std::min(c0.size(1), hidden_size));
                if (c0.size(1) < hidden_size) {
                    auto padding = torch::zeros({c0.size(0), hidden_size - c0.size(1)}, c0.options());
                    c0 = torch::cat({c0, padding}, 1);
                }
            }
        }
        
        // Make sure batch sizes match
        if (input.dim() > 0 && h0.dim() > 0 && input.size(0) != h0.size(0)) {
            int64_t batch_size = std::min(input.size(0), h0.size(0));
            input = input.slice(0, 0, batch_size);
            h0 = h0.slice(0, 0, batch_size);
            c0 = c0.slice(0, 0, batch_size);
        }
        
        // Convert tensors to same dtype if needed
        auto target_dtype = torch::kFloat;
        input = input.to(target_dtype);
        h0 = h0.to(target_dtype);
        c0 = c0.to(target_dtype);
        
        // Apply LSTM cell
        auto result = lstm_cell(input, std::make_tuple(h0, c0));
        
        // Extract outputs
        auto h1 = std::get<0>(result);
        auto c1 = std::get<1>(result);
        
        // Perform some operations on the results to ensure they're used
        auto sum_h = torch::sum(h1);
        auto sum_c = torch::sum(c1);
        auto total_sum = sum_h + sum_c;
        
        // Prevent the compiler from optimizing away the computation
        if (total_sum.item<float>() == -999999.0f) {
            throw std::runtime_error("This should never happen");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}