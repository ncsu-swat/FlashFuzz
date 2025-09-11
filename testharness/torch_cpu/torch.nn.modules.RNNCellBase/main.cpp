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
        
        // Get parameters for RNNCellBase
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        bool bias = true;
        
        // Extract parameters from remaining data
        if (offset + 8 <= Size) {
            std::memcpy(&input_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            std::memcpy(&hidden_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make sure input_size and hidden_size are reasonable
            input_size = std::abs(input_size) % 100 + 1;
            hidden_size = std::abs(hidden_size) % 100 + 1;
            
            // Set bias based on remaining byte if available
            if (offset < Size) {
                bias = Data[offset++] & 0x1;
            }
        } else {
            // Default values if not enough data
            input_size = 10;
            hidden_size = 20;
        }
        
        // Create RNNCellBase
        torch::nn::RNNCellOptions options(input_size, hidden_size);
        options.bias(bias);
        auto rnn_cell = torch::nn::RNNCell(options);
        
        // Create hidden state tensor
        torch::Tensor hidden;
        if (offset < Size) {
            hidden = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default hidden state if no more data
            if (input.dim() > 0 && input.size(0) > 0) {
                hidden = torch::zeros({input.size(0), hidden_size});
            } else {
                hidden = torch::zeros({1, hidden_size});
            }
        }
        
        // Try to reshape input tensor to match expected dimensions if needed
        if (input.dim() == 0) {
            // Scalar input - reshape to [1, input_size]
            input = input.reshape({1, input_size});
        } else if (input.dim() == 1) {
            // 1D input - reshape to [1, min(input.size(0), input_size)]
            int64_t feat_size = std::min(input.size(0), input_size);
            input = input.slice(0, 0, feat_size).reshape({1, feat_size});
            if (feat_size < input_size) {
                input = torch::cat({input, torch::zeros({1, input_size - feat_size})}, 1);
            }
        } else if (input.dim() >= 2) {
            // Multi-dimensional input - reshape to [batch_size, input_size]
            int64_t batch_size = input.size(0);
            int64_t feat_size = std::min(input.size(1), input_size);
            input = input.slice(1, 0, feat_size);
            if (feat_size < input_size) {
                input = torch::cat({input, torch::zeros({batch_size, input_size - feat_size})}, 1);
            }
        }
        
        // Similarly reshape hidden state if needed
        if (hidden.dim() == 0) {
            hidden = hidden.reshape({1, hidden_size});
        } else if (hidden.dim() == 1) {
            int64_t feat_size = std::min(hidden.size(0), hidden_size);
            hidden = hidden.slice(0, 0, feat_size).reshape({1, feat_size});
            if (feat_size < hidden_size) {
                hidden = torch::cat({hidden, torch::zeros({1, hidden_size - feat_size})}, 1);
            }
        } else if (hidden.dim() >= 2) {
            int64_t batch_size = hidden.size(0);
            int64_t feat_size = std::min(hidden.size(1), hidden_size);
            hidden = hidden.slice(1, 0, feat_size);
            if (feat_size < hidden_size) {
                hidden = torch::cat({hidden, torch::zeros({batch_size, hidden_size - feat_size})}, 1);
            }
        }
        
        // Make sure batch sizes match
        if (input.size(0) != hidden.size(0)) {
            int64_t batch_size = std::min(input.size(0), hidden.size(0));
            input = input.slice(0, 0, batch_size);
            hidden = hidden.slice(0, 0, batch_size);
        }
        
        // Convert tensors to same dtype if needed
        if (input.dtype() != hidden.dtype()) {
            if (input.is_floating_point()) {
                hidden = hidden.to(input.dtype());
            } else {
                input = input.to(hidden.dtype());
            }
        }
        
        // Apply the RNN cell
        torch::Tensor output = rnn_cell(input, hidden);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
