#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters from fuzz data
        int64_t input_size = (Data[offset] % 50) + 1;
        offset++;
        int64_t hidden_size = (Data[offset] % 50) + 1;
        offset++;
        bool bias = Data[offset] & 0x1;
        offset++;
        bool use_relu = Data[offset] & 0x1;  // Test different nonlinearities
        offset++;
        
        // Determine batch size from fuzz data
        int64_t batch_size = (Data[offset] % 8) + 1;
        offset++;
        
        // Create RNNCell with options
        // RNNCellBase is abstract; RNNCell is the concrete implementation
        torch::nn::RNNCellOptions options(input_size, hidden_size);
        options.bias(bias);
        if (use_relu) {
            options.nonlinearity(torch::kReLU);
        } else {
            options.nonlinearity(torch::kTanh);
        }
        
        auto rnn_cell = torch::nn::RNNCell(options);
        
        // Create input tensor with proper shape [batch, input_size]
        torch::Tensor input;
        if (offset + 4 <= Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            input = torch::randn({batch_size, input_size});
        }
        
        // Ensure input is floating point
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Reshape input to expected dimensions [batch, input_size]
        try {
            if (input.numel() == 0) {
                input = torch::randn({batch_size, input_size});
            } else if (input.dim() == 0) {
                input = input.expand({batch_size, input_size}).clone();
            } else if (input.dim() == 1) {
                // Expand or truncate to input_size, then add batch dimension
                if (input.size(0) >= input_size) {
                    input = input.slice(0, 0, input_size).unsqueeze(0).expand({batch_size, -1}).clone();
                } else {
                    auto padded = torch::zeros({input_size});
                    padded.slice(0, 0, input.size(0)).copy_(input);
                    input = padded.unsqueeze(0).expand({batch_size, -1}).clone();
                }
            } else {
                // input.dim() >= 2
                int64_t actual_batch = std::min(input.size(0), batch_size);
                input = input.slice(0, 0, actual_batch);
                
                if (input.size(1) >= input_size) {
                    input = input.slice(1, 0, input_size).contiguous();
                } else {
                    auto padded = torch::zeros({actual_batch, input_size});
                    padded.slice(1, 0, input.size(1)).copy_(input.slice(1, 0, input.size(1)));
                    input = padded;
                }
                batch_size = actual_batch;
            }
        } catch (...) {
            // Fallback to random tensor on reshape failure
            input = torch::randn({batch_size, input_size});
        }
        
        // Ensure final shape and type
        if (input.size(0) != batch_size || input.size(1) != input_size) {
            input = torch::randn({batch_size, input_size});
        }
        input = input.to(torch::kFloat32);
        
        // Create hidden state tensor with proper shape [batch, hidden_size]
        torch::Tensor hidden;
        if (offset + 4 <= Size) {
            hidden = fuzzer_utils::createTensor(Data, Size, offset);
            if (!hidden.is_floating_point()) {
                hidden = hidden.to(torch::kFloat32);
            }
            
            try {
                if (hidden.numel() == 0) {
                    hidden = torch::zeros({batch_size, hidden_size});
                } else if (hidden.dim() == 0) {
                    hidden = hidden.expand({batch_size, hidden_size}).clone();
                } else if (hidden.dim() == 1) {
                    if (hidden.size(0) >= hidden_size) {
                        hidden = hidden.slice(0, 0, hidden_size).unsqueeze(0).expand({batch_size, -1}).clone();
                    } else {
                        auto padded = torch::zeros({hidden_size});
                        padded.slice(0, 0, hidden.size(0)).copy_(hidden);
                        hidden = padded.unsqueeze(0).expand({batch_size, -1}).clone();
                    }
                } else {
                    if (hidden.size(0) >= batch_size && hidden.size(1) >= hidden_size) {
                        hidden = hidden.slice(0, 0, batch_size).slice(1, 0, hidden_size).contiguous();
                    } else {
                        hidden = torch::zeros({batch_size, hidden_size});
                    }
                }
            } catch (...) {
                hidden = torch::zeros({batch_size, hidden_size});
            }
        } else {
            hidden = torch::zeros({batch_size, hidden_size});
        }
        
        // Ensure final shape and type
        if (hidden.size(0) != batch_size || hidden.size(1) != hidden_size) {
            hidden = torch::zeros({batch_size, hidden_size});
        }
        hidden = hidden.to(torch::kFloat32);
        
        // Test forward pass with hidden state
        torch::Tensor output = rnn_cell->forward(input, hidden);
        
        // Also test forward without explicit hidden state (uses zeros)
        torch::Tensor output2 = rnn_cell->forward(input);
        
        // Use outputs to prevent optimization
        auto sum1 = output.sum();
        auto sum2 = output2.sum();
        (void)sum1;
        (void)sum2;
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}