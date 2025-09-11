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
        
        // Create weight tensors
        torch::Tensor weight_ih = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor weight_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create bias tensors (optional)
        bool has_biases = offset < Size && (Data[offset++] & 0x1);
        torch::Tensor bias_ih, bias_hh;
        
        if (has_biases) {
            bias_ih = fuzzer_utils::createTensor(Data, Size, offset);
            bias_hh = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Get parameters for RNN
        int64_t hidden_size = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&hidden_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            hidden_size = std::abs(hidden_size) % 64 + 1; // Ensure positive and reasonable size
        } else {
            hidden_size = 10; // Default value
        }
        
        int64_t num_layers = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&num_layers, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_layers = std::abs(num_layers) % 3 + 1; // 1-3 layers is reasonable
        }
        
        bool batch_first = offset < Size && (Data[offset++] & 0x1);
        bool bidirectional = offset < Size && (Data[offset++] & 0x1);
        double dropout = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&dropout, Data + offset, sizeof(double));
            offset += sizeof(double);
            dropout = std::abs(dropout) / 10.0; // Ensure reasonable dropout value
        }
        
        // Create initial hidden state (optional)
        bool has_h0 = offset < Size && (Data[offset++] & 0x1);
        torch::Tensor h0;
        if (has_h0) {
            h0 = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Try different variants of the RNN call
        try {
            // Variant 1: Basic RNN call
            auto rnn = torch::nn::RNN(torch::nn::RNNOptions(input.size(-1), hidden_size)
                                     .num_layers(num_layers)
                                     .batch_first(batch_first)
                                     .bidirectional(bidirectional)
                                     .dropout(dropout)
                                     .nonlinearity(torch::kReLU));
            
            auto output = rnn->forward(input);
            
            // Variant 2: With initial hidden state
            if (has_h0) {
                auto output_with_h0 = rnn->forward(input, h0);
            }
            
            // Variant 3: Direct call to functional API
            if (has_biases) {
                auto output_functional = torch::rnn_relu(
                    input, h0, 
                    {weight_ih, weight_hh, bias_ih, bias_hh}, 
                    has_biases,
                    num_layers, dropout, true, 
                    bidirectional, batch_first);
            } else {
                auto output_functional = torch::rnn_relu(
                    input, h0, 
                    {weight_ih, weight_hh}, 
                    has_biases,
                    num_layers, dropout, true, 
                    bidirectional, batch_first);
            }
            
            // Variant 4: Cell-level operations
            auto rnn_cell = torch::nn::RNNCell(torch::nn::RNNCellOptions(input.size(-1), hidden_size)
                                              .nonlinearity(torch::kReLU));
            
            // Try to get a single input for the cell
            torch::Tensor single_input;
            if (input.dim() > 1) {
                single_input = input.select(0, 0);
            } else {
                single_input = input;
            }
            
            // Create a hidden state for the cell if needed
            torch::Tensor cell_hidden;
            if (has_h0 && h0.dim() > 0) {
                cell_hidden = h0.select(0, 0);
            } else {
                cell_hidden = torch::zeros({hidden_size}, input.options());
            }
            
            auto cell_output = rnn_cell->forward(single_input, cell_hidden);
        } catch (const std::exception& e) {
            // Catch inner exceptions but continue with the fuzzing
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
