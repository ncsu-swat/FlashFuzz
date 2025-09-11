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
        
        // Ensure input has at least 2 dimensions for LSTM
        if (input.dim() < 2) {
            input = input.reshape({1, input.numel()});
        }
        
        // Extract dimensions for LSTM parameters
        int64_t batch_size = input.size(0);
        int64_t input_size = input.size(1);
        
        // Create hidden state tensors (h0, c0)
        int64_t hidden_size = 0;
        
        // Parse hidden size from input data if available
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&hidden_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure hidden_size is positive and reasonable
            hidden_size = std::abs(hidden_size) % 128 + 1;
        } else {
            // Default hidden size if not enough data
            hidden_size = 4;
        }
        
        // Create initial hidden states
        torch::Tensor h0 = torch::zeros({batch_size, hidden_size});
        torch::Tensor c0 = torch::zeros({batch_size, hidden_size});
        
        // Create weight matrices for LSTM cell
        torch::Tensor w_ih = torch::randn({4 * hidden_size, input_size});
        torch::Tensor w_hh = torch::randn({4 * hidden_size, hidden_size});
        torch::Tensor b_ih = torch::randn({4 * hidden_size});
        torch::Tensor b_hh = torch::randn({4 * hidden_size});
        
        // Create hidden state list
        std::vector<torch::Tensor> hx = {h0, c0};
        
        // Apply the LSTM cell
        auto result = torch::lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
        
        // Extract results
        torch::Tensor h_out = std::get<0>(result);
        torch::Tensor c_out = std::get<1>(result);
        
        // Try different input shapes if there's more data
        if (offset + 4 < Size) {
            // Create another input with different shape
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape if needed
            if (input2.dim() < 2) {
                input2 = input2.reshape({1, input2.numel()});
            }
            
            // Adjust input2 to match input_size if needed
            if (input2.size(1) != input_size) {
                input2 = input2.reshape({input2.size(0), input_size});
            }
            
            // Try with the output of the previous call as initial state
            std::vector<torch::Tensor> hx2 = {h_out, c_out};
            auto result2 = torch::lstm_cell(input2, hx2, w_ih, w_hh, b_ih, b_hh);
        }
        
        // Try without bias if there's more data
        if (offset + 1 < Size) {
            uint8_t use_bias = Data[offset++];
            
            if (use_bias % 2 == 0) {
                // Without bias
                auto result3 = torch::lstm_cell(input, hx, w_ih, w_hh);
            } else {
                // With only input-hidden bias
                auto result3 = torch::lstm_cell(input, hx, w_ih, w_hh, b_ih);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
