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
            // If we don't have enough data for h0, create a default one
            h0 = torch::zeros({input.size(0), 10});
        }
        
        // Extract parameters for RNNCell
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        
        // Get input_size from input tensor if possible
        if (input.dim() >= 2) {
            input_size = input.size(-1);
        } else if (input.dim() == 1) {
            input_size = input.size(0);
        } else {
            input_size = 1;
        }
        
        // Get hidden_size from h0 tensor if possible
        if (h0.dim() >= 1) {
            hidden_size = h0.size(-1);
        } else {
            hidden_size = 10; // Default value
        }
        
        // Ensure input_size and hidden_size are positive
        input_size = std::max(int64_t(1), input_size);
        hidden_size = std::max(int64_t(1), hidden_size);
        
        // Create RNNCell using regular RNNCell since quantized dynamic version is not available
        torch::nn::RNNCell rnn_model(
            torch::nn::RNNCellOptions(input_size, hidden_size)
        );
        
        // Reshape input if needed to match expected format [batch_size, input_size]
        if (input.dim() == 0) {
            input = input.reshape({1, 1});
        } else if (input.dim() == 1) {
            input = input.reshape({1, input.size(0)});
        } else if (input.dim() > 2) {
            // Flatten all dimensions except the last one
            std::vector<int64_t> new_shape = {-1, input.size(-1)};
            input = input.reshape(new_shape);
        }
        
        // Reshape h0 if needed to match expected format [batch_size, hidden_size]
        if (h0.dim() == 0) {
            h0 = h0.reshape({1, 1});
        } else if (h0.dim() == 1) {
            h0 = h0.reshape({1, h0.size(0)});
        } else if (h0.dim() > 2) {
            // Flatten all dimensions except the last one
            std::vector<int64_t> new_shape = {-1, h0.size(-1)};
            h0 = h0.reshape(new_shape);
        }
        
        // Make sure batch sizes match
        if (input.size(0) != h0.size(0)) {
            int64_t batch_size = std::min(input.size(0), h0.size(0));
            input = input.slice(0, 0, batch_size);
            h0 = h0.slice(0, 0, batch_size);
        }
        
        // Make sure input_size matches model's input_size
        if (input.size(1) != input_size) {
            if (input.size(1) > input_size) {
                input = input.slice(1, 0, input_size);
            } else {
                // Pad with zeros
                torch::Tensor padding = torch::zeros({input.size(0), input_size - input.size(1)});
                input = torch::cat({input, padding}, 1);
            }
        }
        
        // Make sure hidden_size matches model's hidden_size
        if (h0.size(1) != hidden_size) {
            if (h0.size(1) > hidden_size) {
                h0 = h0.slice(1, 0, hidden_size);
            } else {
                // Pad with zeros
                torch::Tensor padding = torch::zeros({h0.size(0), hidden_size - h0.size(1)});
                h0 = torch::cat({h0, padding}, 1);
            }
        }
        
        // Convert tensors to float if they're not already
        if (input.scalar_type() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        if (h0.scalar_type() != torch::kFloat) {
            h0 = h0.to(torch::kFloat);
        }
        
        // Forward pass
        torch::Tensor output = rnn_model(input, h0);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}