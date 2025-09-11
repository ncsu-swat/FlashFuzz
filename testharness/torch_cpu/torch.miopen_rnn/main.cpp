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
        
        if (Size < 10) return 0; // Need minimum data for basic parameters
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse RNN parameters from the remaining data
        if (offset + 8 >= Size) return 0;
        
        // Parse mode (RNN, LSTM, GRU)
        uint8_t mode_byte = Data[offset++] % 3;
        int64_t mode = static_cast<int64_t>(mode_byte);
        
        // Parse input_size
        uint8_t input_size_byte = Data[offset++];
        int64_t input_size = static_cast<int64_t>(input_size_byte % 32 + 1);
        
        // Parse hidden_size
        uint8_t hidden_size_byte = Data[offset++];
        int64_t hidden_size = static_cast<int64_t>(hidden_size_byte % 32 + 1);
        
        // Parse num_layers
        uint8_t num_layers_byte = Data[offset++];
        int64_t num_layers = static_cast<int64_t>(num_layers_byte % 4 + 1);
        
        // Parse batch_first
        bool batch_first = (Data[offset++] % 2) == 1;
        
        // Parse dropout
        double dropout = static_cast<double>(Data[offset++]) / 255.0;
        
        // Parse train
        bool train = (Data[offset++] % 2) == 1;
        
        // Parse bidirectional
        bool bidirectional = (Data[offset++] % 2) == 1;
        
        // Create weight tensors as TensorList
        std::vector<torch::Tensor> weight_list;
        if (offset < Size) {
            torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
            weight_list.push_back(weight);
        } else {
            // Create a default weight tensor if we don't have enough data
            int64_t num_directions = bidirectional ? 2 : 1;
            int64_t weight_size = 0;
            
            // Different weight sizes for different RNN types
            if (mode == 0) { // RNN
                weight_size = num_directions * (hidden_size * input_size + hidden_size * hidden_size + 2 * hidden_size);
            } else if (mode == 1) { // LSTM
                weight_size = num_directions * 4 * (hidden_size * input_size + hidden_size * hidden_size + 2 * hidden_size);
            } else { // GRU
                weight_size = num_directions * 3 * (hidden_size * input_size + hidden_size * hidden_size + 2 * hidden_size);
            }
            
            torch::Tensor weight = torch::ones({weight_size}, torch::kFloat);
            weight_list.push_back(weight);
        }
        
        // Parse weight_stride0 as int64_t
        int64_t weight_stride0 = 1;
        if (offset < Size) {
            weight_stride0 = static_cast<int64_t>(Data[offset++] % 10 + 1);
        }
        
        // Create hx tensor
        torch::Tensor hx;
        if (offset < Size) {
            hx = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Default hx
            int64_t num_directions = bidirectional ? 2 : 1;
            hx = torch::zeros({num_layers * num_directions, 1, hidden_size}, torch::kFloat);
        }
        
        // Create optional cx tensor
        std::optional<torch::Tensor> cx;
        if (offset < Size && mode == 1) { // Only for LSTM
            cx = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Create batch_sizes
        std::vector<int64_t> batch_sizes = {1};
        
        // Create optional dropout_state
        std::optional<torch::Tensor> dropout_state;
        if (offset < Size && dropout > 0.0) {
            dropout_state = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Try to call miopen_rnn
        try {
            auto result = torch::miopen_rnn(
                input,
                weight_list,
                weight_stride0,
                hx,
                cx,
                mode,
                hidden_size,
                num_layers,
                batch_first,
                dropout,
                train,
                bidirectional,
                batch_sizes,
                dropout_state
            );
        } catch (const c10::Error& e) {
            // Expected PyTorch errors are fine
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
