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
        
        // Extract parameters for LSTM from the remaining data
        uint8_t hidden_size = 0;
        uint8_t num_layers = 0;
        bool bidirectional = false;
        bool batch_first = false;
        float dropout = 0.0f;
        
        if (offset + 4 <= Size) {
            hidden_size = Data[offset++] % 32 + 1; // Ensure non-zero hidden size
            num_layers = Data[offset++] % 4 + 1;   // 1-4 layers
            bidirectional = Data[offset++] % 2;    // 0 or 1
            batch_first = Data[offset++] % 2;      // 0 or 1
            
            if (offset + sizeof(float) <= Size) {
                float raw_dropout;
                std::memcpy(&raw_dropout, Data + offset, sizeof(float));
                offset += sizeof(float);
                
                // Ensure dropout is between 0 and 1
                dropout = std::abs(raw_dropout) / (std::abs(raw_dropout) + 1.0f);
            }
        }
        
        // Create LSTM module
        torch::nn::LSTM lstm_options = torch::nn::LSTM(
            torch::nn::LSTMOptions(input.size(-1), hidden_size)
                .num_layers(num_layers)
                .dropout(dropout)
                .bidirectional(bidirectional)
                .batch_first(batch_first)
        );
        
        // Prepare initial hidden state (h0, c0)
        int64_t batch_size = batch_first ? input.size(0) : input.size(1);
        int64_t num_directions = bidirectional ? 2 : 1;
        
        torch::Tensor h0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
        torch::Tensor c0 = torch::zeros({num_layers * num_directions, batch_size, hidden_size});
        
        // Run LSTM forward
        auto output = lstm_options->forward(input, std::make_tuple(h0, c0));
        
        // Extract output and hidden states
        torch::Tensor output_seq = std::get<0>(output);
        auto hidden_states = std::get<1>(output);
        torch::Tensor h_n = std::get<0>(hidden_states);
        torch::Tensor c_n = std::get<1>(hidden_states);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output_seq.sum() + h_n.sum() + c_n.sum();
        
        // Try different input shapes if there's more data
        if (offset + 10 < Size) {
            // Create another input tensor with different shape
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Try to run LSTM with this new input
            try {
                auto output2 = lstm_options->forward(input2, std::make_tuple(h0, c0));
                auto sum2 = std::get<0>(output2).sum();
            } catch (const std::exception&) {
                // Ignore exceptions from the second run
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