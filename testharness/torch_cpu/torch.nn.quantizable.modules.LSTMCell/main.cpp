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
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create hidden state tensors
        torch::Tensor h0, c0;
        if (offset < Size - 5) {
            h0 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            h0 = torch::zeros({input.size(0), 20});
        }
        
        if (offset < Size - 5) {
            c0 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            c0 = torch::zeros({input.size(0), 20});
        }
        
        // Get parameters for LSTMCell
        int64_t input_size = 0;
        int64_t hidden_size = 0;
        bool bias = true;
        
        if (offset + 8 <= Size) {
            std::memcpy(&input_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&hidden_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make sure input_size and hidden_size are reasonable
            input_size = std::abs(input_size) % 100 + 1;
            hidden_size = std::abs(hidden_size) % 100 + 1;
            
            if (offset < Size) {
                bias = Data[offset++] & 0x1;
            }
        } else {
            // Default values if not enough data
            input_size = 10;
            hidden_size = 20;
        }
        
        // Reshape input tensor if needed to match input_size
        if (input.dim() == 0) {
            input = input.reshape({1, input_size});
        } else if (input.dim() == 1) {
            input = input.reshape({1, input.size(0)});
            if (input.size(1) != input_size) {
                input = input.slice(1, 0, std::min(input.size(1), static_cast<int64_t>(input_size)));
                if (input.size(1) < input_size) {
                    auto padding = torch::zeros({input.size(0), input_size - input.size(1)}, input.options());
                    input = torch::cat({input, padding}, 1);
                }
            }
        } else if (input.dim() >= 2) {
            if (input.size(1) != input_size) {
                input = input.slice(1, 0, std::min(input.size(1), static_cast<int64_t>(input_size)));
                if (input.size(1) < input_size) {
                    auto padding = torch::zeros({input.size(0), input_size - input.size(1)}, input.options());
                    input = torch::cat({input, padding}, 1);
                }
            }
        }
        
        // Reshape hidden state tensors if needed to match batch_size and hidden_size
        int64_t batch_size = input.size(0);
        
        if (h0.dim() == 0) {
            h0 = torch::zeros({batch_size, hidden_size});
        } else if (h0.sizes() != std::vector<int64_t>{batch_size, hidden_size}) {
            h0 = torch::zeros({batch_size, hidden_size});
        }
        
        if (c0.dim() == 0) {
            c0 = torch::zeros({batch_size, hidden_size});
        } else if (c0.sizes() != std::vector<int64_t>{batch_size, hidden_size}) {
            c0 = torch::zeros({batch_size, hidden_size});
        }
        
        // Convert tensors to float if needed
        if (input.scalar_type() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        if (h0.scalar_type() != torch::kFloat) {
            h0 = h0.to(torch::kFloat);
        }
        if (c0.scalar_type() != torch::kFloat) {
            c0 = c0.to(torch::kFloat);
        }
        
        // Create LSTMCell
        torch::nn::LSTMCell lstm_cell(
            torch::nn::LSTMCellOptions(input_size, hidden_size).bias(bias)
        );
        
        // Apply LSTMCell
        auto result = lstm_cell(input, std::make_tuple(h0, c0));
        
        // Get output tensors
        auto h1 = std::get<0>(result);
        auto c1 = std::get<1>(result);
        
        // Test with different inputs if we have more data
        if (offset < Size) {
            auto input2 = torch::randn({batch_size, input_size});
            auto result2 = lstm_cell(input2, std::make_tuple(h1, c1));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
