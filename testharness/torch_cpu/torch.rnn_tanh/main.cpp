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
            int64_t batch_size = input.size(1);
            int64_t hidden_size = 10; // Arbitrary hidden size
            h0 = torch::zeros({1, batch_size, hidden_size});
        }
        
        // Create weight parameters
        torch::Tensor weight_ih, weight_hh, bias_ih, bias_hh;
        
        if (offset < Size) {
            weight_ih = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            int64_t input_size = input.size(2);
            int64_t hidden_size = h0.size(2);
            weight_ih = torch::randn({hidden_size, input_size});
        }
        
        if (offset < Size) {
            weight_hh = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            int64_t hidden_size = h0.size(2);
            weight_hh = torch::randn({hidden_size, hidden_size});
        }
        
        if (offset < Size) {
            bias_ih = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            int64_t hidden_size = h0.size(2);
            bias_ih = torch::randn({hidden_size});
        }
        
        if (offset < Size) {
            bias_hh = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            int64_t hidden_size = h0.size(2);
            bias_hh = torch::randn({hidden_size});
        }
        
        // Get num_layers from remaining data if available
        int64_t num_layers = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&num_layers, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_layers = std::abs(num_layers) % 3 + 1; // Limit to 1-3 layers
        }
        
        // Get dropout value from remaining data if available
        double dropout = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&dropout, Data + offset, sizeof(double));
            offset += sizeof(double);
            dropout = std::abs(dropout) / 10.0; // Ensure dropout is between 0 and 0.1
        }
        
        // Get bidirectional flag from remaining data if available
        bool bidirectional = false;
        if (offset < Size) {
            bidirectional = Data[offset++] % 2 == 1;
        }
        
        // Get batch_first flag from remaining data if available
        bool batch_first = false;
        if (offset < Size) {
            batch_first = Data[offset++] % 2 == 1;
        }
        
        // Get has_biases flag from remaining data if available
        bool has_biases = true;
        if (offset < Size) {
            has_biases = Data[offset++] % 2 == 1;
        }
        
        // Get train flag from remaining data if available
        bool train = false;
        if (offset < Size) {
            train = Data[offset++] % 2 == 1;
        }
        
        try {
            // Apply RNN tanh operation
            auto result = torch::rnn_tanh(
                input,
                h0,
                {weight_ih, weight_hh, bias_ih, bias_hh},
                has_biases,
                num_layers,
                dropout,
                train,
                bidirectional,
                batch_first
            );
            
            // Access the output to ensure computation is performed
            auto output = std::get<0>(result);
            auto h_n = std::get<1>(result);
            
            // Perform some operation on the result to ensure it's used
            auto sum = output.sum() + h_n.sum();
            if (sum.item<float>() == 0.0f) {
                // Just to use the result
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and part of the fuzzing process
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}