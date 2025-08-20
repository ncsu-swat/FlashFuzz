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
            // Create default h0 if we don't have enough data
            h0 = torch::zeros({1, 10});
        }
        
        if (offset < Size) {
            c0 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create default c0 if we don't have enough data
            c0 = torch::zeros({1, 10});
        }
        
        // Extract parameters for LSTMCell
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        bool bias = true;
        
        // Parse input_size
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&input_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            input_size = std::abs(input_size) % 100 + 1; // Ensure positive and reasonable size
        } else {
            // Default value if not enough data
            input_size = 10;
        }
        
        // Parse hidden_size
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&hidden_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            hidden_size = std::abs(hidden_size) % 100 + 1; // Ensure positive and reasonable size
        } else {
            // Default value if not enough data
            hidden_size = 20;
        }
        
        // Parse bias flag
        if (offset < Size) {
            bias = Data[offset++] & 0x1; // Use lowest bit to determine bias
        }
        
        // Create LSTMCell using regular nn module
        torch::nn::LSTMCell lstm_cell(
            torch::nn::LSTMCellOptions(input_size, hidden_size).bias(bias)
        );
        
        // Try to reshape input tensor to match expected input shape if needed
        if (input.dim() > 0) {
            // For LSTMCell, input should be [batch_size, input_size]
            // Try to reshape to match this requirement
            int64_t batch_size = 1;
            if (input.dim() >= 2) {
                batch_size = input.size(0);
            }
            
            try {
                input = input.reshape({batch_size, input_size});
            } catch (const std::exception&) {
                // If reshape fails, create a new tensor with the right shape
                input = torch::ones({batch_size, input_size});
            }
        } else {
            // If input is a scalar, create a proper tensor
            input = torch::ones({1, input_size});
        }
        
        // Try to reshape h0 and c0 to match expected shape
        try {
            int64_t batch_size = input.size(0);
            h0 = h0.reshape({batch_size, hidden_size});
            c0 = c0.reshape({batch_size, hidden_size});
        } catch (const std::exception&) {
            // If reshape fails, create new tensors with the right shape
            int64_t batch_size = input.size(0);
            h0 = torch::zeros({batch_size, hidden_size});
            c0 = torch::zeros({batch_size, hidden_size});
        }
        
        // Apply the LSTMCell operation
        auto result = lstm_cell(input, std::make_tuple(h0, c0));
        
        // Extract the output tensors
        auto h1 = std::get<0>(result);
        auto c1 = std::get<1>(result);
        
        // Test with different input variations
        try {
            // Test with different batch sizes
            auto input2 = torch::randn({2, input_size});
            auto h0_2 = torch::zeros({2, hidden_size});
            auto c0_2 = torch::zeros({2, hidden_size});
            auto result2 = lstm_cell(input2, std::make_tuple(h0_2, c0_2));
        } catch (const std::exception&) {
            // Different batch size might fail, that's okay for fuzzing
        }
        
        // Test gradient computation
        try {
            input.requires_grad_(true);
            h0.requires_grad_(true);
            c0.requires_grad_(true);
            
            auto grad_result = lstm_cell(input, std::make_tuple(h0, c0));
            auto loss = std::get<0>(grad_result).sum();
            loss.backward();
        } catch (const std::exception&) {
            // Gradient computation might fail, that's okay for fuzzing
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}