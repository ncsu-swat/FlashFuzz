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
        torch::Tensor h0;
        if (offset < Size) {
            h0 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for h0, create a compatible one
            if (input.dim() > 0 && input.size(0) > 0) {
                h0 = torch::zeros({input.size(0), 10}, torch::kFloat);
            } else {
                h0 = torch::zeros({1, 10}, torch::kFloat);
            }
        }
        
        // Extract parameters for RNNCell
        int64_t input_size = 10;
        int64_t hidden_size = 20;
        
        // If we have input tensor with valid dimensions, use them
        if (input.dim() >= 2) {
            input_size = input.size(1);
        }
        
        // If we have hidden state with valid dimensions, use them
        if (h0.dim() >= 2) {
            hidden_size = h0.size(1);
        }
        
        // Create RNNCell (using regular RNNCell since quantized dynamic version is not available)
        torch::nn::RNNCell cell(
            torch::nn::RNNCellOptions(input_size, hidden_size)
        );
        
        // Ensure input has correct shape for RNNCell
        if (input.dim() == 0) {
            // Scalar tensor - reshape to [1, input_size]
            input = input.reshape({1, input_size});
        } else if (input.dim() == 1) {
            // 1D tensor - reshape to [1, size]
            input = input.reshape({1, input.size(0)});
        } else if (input.dim() > 2) {
            // Higher dimensional tensor - reshape to [batch_size, input_size]
            input = input.reshape({input.size(0), -1});
        }
        
        // Ensure hidden state has correct shape
        if (h0.dim() == 0) {
            h0 = h0.reshape({1, hidden_size});
        } else if (h0.dim() == 1) {
            h0 = h0.reshape({1, h0.size(0)});
        } else if (h0.dim() > 2) {
            h0 = h0.reshape({h0.size(0), -1});
        }
        
        // Make sure input and h0 have compatible batch sizes
        if (input.size(0) != h0.size(0)) {
            // Resize one of them to match
            if (input.size(0) > h0.size(0)) {
                h0 = h0.repeat({input.size(0) / h0.size(0) + 1, 1});
                h0 = h0.slice(0, 0, input.size(0));
            } else {
                input = input.repeat({h0.size(0) / input.size(0) + 1, 1});
                input = input.slice(0, 0, h0.size(0));
            }
        }
        
        // Make sure input and h0 have correct feature dimensions
        if (input.size(1) != input_size) {
            input = input.slice(1, 0, std::min(input.size(1), input_size));
            if (input.size(1) < input_size) {
                auto padding = torch::zeros({input.size(0), input_size - input.size(1)}, input.options());
                input = torch::cat({input, padding}, 1);
            }
        }
        
        if (h0.size(1) != hidden_size) {
            h0 = h0.slice(1, 0, std::min(h0.size(1), hidden_size));
            if (h0.size(1) < hidden_size) {
                auto padding = torch::zeros({h0.size(0), hidden_size - h0.size(1)}, h0.options());
                h0 = torch::cat({h0, padding}, 1);
            }
        }
        
        // Convert tensors to float if needed
        if (input.scalar_type() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        if (h0.scalar_type() != torch::kFloat) {
            h0 = h0.to(torch::kFloat);
        }
        
        // Forward pass
        torch::Tensor output = cell(input, h0);
        
        // Try different bias configurations if we have more data
        if (offset + 1 < Size) {
            bool use_bias = Data[offset++] % 2 == 0;
            torch::nn::RNNCell cell2(
                torch::nn::RNNCellOptions(input_size, hidden_size).bias(use_bias)
            );
            torch::Tensor output2 = cell2(input, h0);
        }
        
        // Try different nonlinearity if we have more data
        if (offset + 1 < Size) {
            std::string nonlinearity = (Data[offset++] % 2 == 0) ? "tanh" : "relu";
            torch::nn::RNNCell cell3(
                torch::nn::RNNCellOptions(input_size, hidden_size).nonlinearity(nonlinearity)
            );
            torch::Tensor output3 = cell3(input, h0);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}