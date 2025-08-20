#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least some bytes left for RNN parameters
        if (offset + 8 >= Size) {
            return 0;
        }
        
        // Parse RNN parameters from the input data
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        int64_t num_layers = 0;
        bool bias = false;
        bool batch_first = false;
        double dropout = 0.0;
        bool bidirectional = false;
        
        // Extract input_size (1-100)
        if (offset < Size) {
            input_size = (Data[offset++] % 100) + 1;
        }
        
        // Extract hidden_size (1-100)
        if (offset < Size) {
            hidden_size = (Data[offset++] % 100) + 1;
        }
        
        // Extract num_layers (1-3)
        if (offset < Size) {
            num_layers = (Data[offset++] % 3) + 1;
        }
        
        // Extract bias flag
        if (offset < Size) {
            bias = Data[offset++] % 2 == 0;
        }
        
        // Extract batch_first flag
        if (offset < Size) {
            batch_first = Data[offset++] % 2 == 0;
        }
        
        // Extract dropout (0.0-0.9)
        if (offset + sizeof(double) <= Size) {
            double raw_dropout;
            std::memcpy(&raw_dropout, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure dropout is between 0 and 0.9
            dropout = std::abs(raw_dropout);
            dropout = dropout - std::floor(dropout); // Get fractional part
            if (dropout > 0.9) dropout = 0.9;
        }
        
        // Extract bidirectional flag
        if (offset < Size) {
            bidirectional = Data[offset++] % 2 == 0;
        }
        
        // Choose RNN mode based on input data
        std::string mode = "RNN_TANH";
        if (offset < Size) {
            uint8_t mode_selector = Data[offset++] % 3;
            switch (mode_selector) {
                case 0: mode = "RNN_TANH"; break;
                case 1: mode = "RNN_RELU"; break;
                case 2: mode = "LSTM"; break;
            }
        }
        
        // Create RNN module
        torch::nn::RNNOptions rnn_options = 
            torch::nn::RNNOptions(input_size, hidden_size)
                .num_layers(num_layers)
                .bias(bias)
                .batch_first(batch_first)
                .dropout(dropout)
                .bidirectional(bidirectional);
        
        torch::nn::RNN rnn_module = nullptr;
        
        if (mode == "RNN_TANH") {
            rnn_module = torch::nn::RNN(rnn_options.nonlinearity(torch::kTanh));
        } else if (mode == "RNN_RELU") {
            rnn_module = torch::nn::RNN(rnn_options.nonlinearity(torch::kReLU));
        } else if (mode == "LSTM") {
            auto lstm_options = 
                torch::nn::LSTMOptions(input_size, hidden_size)
                    .num_layers(num_layers)
                    .bias(bias)
                    .batch_first(batch_first)
                    .dropout(dropout)
                    .bidirectional(bidirectional);
            
            torch::nn::LSTM lstm_module(lstm_options);
            
            // Try to reshape input tensor to match LSTM requirements if needed
            try {
                // Reshape input tensor if necessary to match expected dimensions
                // For LSTM: [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
                auto input_sizes = input_tensor.sizes();
                
                if (input_sizes.size() < 2) {
                    // Add dimensions if needed
                    if (input_sizes.size() == 0) {
                        // Scalar tensor, reshape to [1, 1, input_size]
                        input_tensor = input_tensor.reshape({1, 1, input_size});
                    } else if (input_sizes.size() == 1) {
                        // 1D tensor, reshape to [1, 1, min(input_size, dim0)]
                        int64_t actual_input_size = std::min(input_size, input_tensor.size(0));
                        input_tensor = input_tensor.slice(0, 0, actual_input_size).reshape({1, 1, actual_input_size});
                    }
                } else if (input_sizes.size() == 2) {
                    // 2D tensor, add a dimension
                    int64_t dim0 = input_tensor.size(0);
                    int64_t dim1 = input_tensor.size(1);
                    
                    if (batch_first) {
                        // [batch, seq_len] -> [batch, seq_len, input_size]
                        input_tensor = input_tensor.reshape({dim0, dim1, 1});
                    } else {
                        // [seq_len, batch] -> [seq_len, batch, input_size]
                        input_tensor = input_tensor.reshape({dim0, dim1, 1});
                    }
                }
                
                // Ensure the last dimension matches input_size
                auto reshaped_sizes = input_tensor.sizes();
                int64_t last_dim = reshaped_sizes.size() - 1;
                
                if (last_dim >= 0 && reshaped_sizes[last_dim] != input_size) {
                    // Create a new tensor with the correct last dimension
                    std::vector<int64_t> new_sizes(reshaped_sizes.begin(), reshaped_sizes.end());
                    new_sizes[last_dim] = input_size;
                    
                    // Create a new tensor with the right shape
                    torch::Tensor new_input = torch::zeros(new_sizes, input_tensor.options());
                    
                    // Copy data from the original tensor, up to the minimum size
                    int64_t copy_size = std::min(reshaped_sizes[last_dim], input_size);
                    if (copy_size > 0) {
                        // Create slices and copy
                        torch::Tensor src_slice = input_tensor.slice(last_dim, 0, copy_size);
                        torch::Tensor dst_slice = new_input.slice(last_dim, 0, copy_size);
                        dst_slice.copy_(src_slice);
                    }
                    
                    input_tensor = new_input;
                }
                
                // Create h0 (initial hidden state)
                int64_t num_directions = bidirectional ? 2 : 1;
                torch::Tensor h0 = torch::zeros({num_layers * num_directions, 
                                                batch_first ? input_tensor.size(0) : input_tensor.size(1), 
                                                hidden_size});
                
                // Create c0 (initial cell state for LSTM)
                torch::Tensor c0 = torch::zeros_like(h0);
                
                // Forward pass
                auto output = lstm_module->forward(input_tensor, std::make_tuple(h0, c0));
            } catch (const std::exception& e) {
                // Catch and ignore exceptions from LSTM forward pass
            }
            
            return 0;
        }
        
        if (rnn_module) {
            try {
                // Try to reshape input tensor to match RNN requirements if needed
                auto input_sizes = input_tensor.sizes();
                
                if (input_sizes.size() < 2) {
                    // Add dimensions if needed
                    if (input_sizes.size() == 0) {
                        // Scalar tensor, reshape to [1, 1, input_size]
                        input_tensor = input_tensor.reshape({1, 1, input_size});
                    } else if (input_sizes.size() == 1) {
                        // 1D tensor, reshape to [1, 1, min(input_size, dim0)]
                        int64_t actual_input_size = std::min(input_size, input_tensor.size(0));
                        input_tensor = input_tensor.slice(0, 0, actual_input_size).reshape({1, 1, actual_input_size});
                    }
                } else if (input_sizes.size() == 2) {
                    // 2D tensor, add a dimension
                    int64_t dim0 = input_tensor.size(0);
                    int64_t dim1 = input_tensor.size(1);
                    
                    if (batch_first) {
                        // [batch, seq_len] -> [batch, seq_len, input_size]
                        input_tensor = input_tensor.reshape({dim0, dim1, 1});
                    } else {
                        // [seq_len, batch] -> [seq_len, batch, input_size]
                        input_tensor = input_tensor.reshape({dim0, dim1, 1});
                    }
                }
                
                // Ensure the last dimension matches input_size
                auto reshaped_sizes = input_tensor.sizes();
                int64_t last_dim = reshaped_sizes.size() - 1;
                
                if (last_dim >= 0 && reshaped_sizes[last_dim] != input_size) {
                    // Create a new tensor with the correct last dimension
                    std::vector<int64_t> new_sizes(reshaped_sizes.begin(), reshaped_sizes.end());
                    new_sizes[last_dim] = input_size;
                    
                    // Create a new tensor with the right shape
                    torch::Tensor new_input = torch::zeros(new_sizes, input_tensor.options());
                    
                    // Copy data from the original tensor, up to the minimum size
                    int64_t copy_size = std::min(reshaped_sizes[last_dim], input_size);
                    if (copy_size > 0) {
                        // Create slices and copy
                        torch::Tensor src_slice = input_tensor.slice(last_dim, 0, copy_size);
                        torch::Tensor dst_slice = new_input.slice(last_dim, 0, copy_size);
                        dst_slice.copy_(src_slice);
                    }
                    
                    input_tensor = new_input;
                }
                
                // Create h0 (initial hidden state)
                int64_t num_directions = bidirectional ? 2 : 1;
                torch::Tensor h0 = torch::zeros({num_layers * num_directions, 
                                                batch_first ? input_tensor.size(0) : input_tensor.size(1), 
                                                hidden_size});
                
                // Forward pass
                auto output = rnn_module->forward(input_tensor, h0);
            } catch (const std::exception& e) {
                // Catch and ignore exceptions from RNN forward pass
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