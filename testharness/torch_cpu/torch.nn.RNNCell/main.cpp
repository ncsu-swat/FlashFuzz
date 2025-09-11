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
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
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
                h0 = torch::zeros({input.size(0), 10}, input.options());
            } else {
                h0 = torch::zeros({1, 10}, input.options());
            }
        }
        
        // Extract parameters for RNNCell
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        bool bias = true;
        bool nonlinearity_tanh = true;
        
        // Parse input_size
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&input_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            input_size = std::abs(input_size) % 100 + 1; // Ensure positive and reasonable
        } else {
            // Default value if not enough data
            input_size = 10;
        }
        
        // Parse hidden_size
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&hidden_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            hidden_size = std::abs(hidden_size) % 100 + 1; // Ensure positive and reasonable
        } else {
            // Default value if not enough data
            hidden_size = 20;
        }
        
        // Parse bias flag
        if (offset < Size) {
            bias = Data[offset++] & 0x1; // Use lowest bit to determine bias
        }
        
        // Parse nonlinearity type
        if (offset < Size) {
            nonlinearity_tanh = Data[offset++] & 0x1; // Use lowest bit to determine nonlinearity
        }
        
        // Create RNNCell
        torch::nn::RNNCellOptions options(input_size, hidden_size);
        options.bias(bias);
        options.nonlinearity(nonlinearity_tanh ? torch::nn::RNNCellOptions::Nonlinearity::Tanh : torch::nn::RNNCellOptions::Nonlinearity::ReLU);
        
        torch::nn::RNNCell cell(options);
        
        // Reshape input tensor if needed to match expected dimensions
        if (input.dim() < 2) {
            if (input.dim() == 0) {
                input = input.reshape({1, 1});
            } else {
                input = input.reshape({1, input.size(0)});
            }
        }
        
        // Reshape h0 to match expected dimensions
        if (h0.dim() < 2) {
            if (h0.dim() == 0) {
                h0 = h0.reshape({1, 1});
            } else {
                h0 = h0.reshape({1, h0.size(0)});
            }
        }
        
        // Try to make dimensions compatible
        if (input.size(1) != input_size) {
            if (input.dim() >= 2) {
                input = input.reshape({input.size(0), input_size});
            }
        }
        
        if (h0.size(1) != hidden_size) {
            if (h0.dim() >= 2) {
                h0 = h0.reshape({h0.size(0), hidden_size});
            }
        }
        
        // Ensure batch sizes match
        if (input.size(0) != h0.size(0)) {
            int64_t batch_size = std::min(input.size(0), h0.size(0));
            if (batch_size > 0) {
                input = input.slice(0, 0, batch_size);
                h0 = h0.slice(0, 0, batch_size);
            } else {
                // Handle edge case where one dimension is 0
                batch_size = std::max(input.size(0), h0.size(0));
                if (input.size(0) == 0) {
                    input = torch::zeros({batch_size, input_size}, input.options());
                }
                if (h0.size(0) == 0) {
                    h0 = torch::zeros({batch_size, hidden_size}, h0.options());
                }
            }
        }
        
        // Apply RNNCell
        torch::Tensor output = cell(input, h0);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        // Prevent optimization from removing the computation
        if (sum.item<float>() == -1.0f) {
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
