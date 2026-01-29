#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least some data to proceed
        if (Size < 6) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract dimensions from fuzzer data
        int64_t batch_size = (Data[offset++] % 8) + 1;    // 1 to 8
        int64_t input_size = (Data[offset++] % 32) + 1;   // 1 to 32
        int64_t hidden_size = (Data[offset++] % 32) + 1;  // 1 to 32
        bool use_bias = (Data[offset++] % 2) == 0;
        
        // Create input tensor with correct shape: (batch, input_size)
        torch::Tensor input = torch::zeros({batch_size, input_size});
        
        // Fill input tensor with fuzzer data
        size_t input_elements = batch_size * input_size;
        if (offset + input_elements * sizeof(float) <= Size) {
            float* input_data = input.data_ptr<float>();
            for (size_t i = 0; i < input_elements && offset + sizeof(float) <= Size; i++) {
                float val;
                memcpy(&val, Data + offset, sizeof(float));
                offset += sizeof(float);
                // Clamp to reasonable range to avoid NaN/Inf issues
                if (std::isnan(val) || std::isinf(val)) {
                    val = 0.0f;
                }
                val = std::max(-10.0f, std::min(10.0f, val));
                input_data[i] = val;
            }
        }
        
        // Create GRUCell with the specified dimensions
        torch::nn::GRUCellOptions options(input_size, hidden_size);
        options.bias(use_bias);
        
        auto gru_cell = torch::nn::GRUCell(options);
        
        // Decide whether to provide initial hidden state
        torch::Tensor hx;
        bool use_initial_hx = (offset < Size) && (Data[offset++] % 2 == 0);
        
        if (use_initial_hx) {
            // Create hidden state tensor with correct shape: (batch, hidden_size)
            hx = torch::zeros({batch_size, hidden_size});
            
            // Fill hx tensor with fuzzer data
            size_t hx_elements = batch_size * hidden_size;
            if (offset + hx_elements * sizeof(float) <= Size) {
                float* hx_data = hx.data_ptr<float>();
                for (size_t i = 0; i < hx_elements && offset + sizeof(float) <= Size; i++) {
                    float val;
                    memcpy(&val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    if (std::isnan(val) || std::isinf(val)) {
                        val = 0.0f;
                    }
                    val = std::max(-10.0f, std::min(10.0f, val));
                    hx_data[i] = val;
                }
            }
        }
        
        // Forward pass - GRUCell can take optional hidden state
        torch::Tensor output;
        if (use_initial_hx) {
            output = gru_cell->forward(input, hx);
        } else {
            // When hx is not provided, GRUCell initializes it to zeros internally
            output = gru_cell->forward(input);
        }
        
        // Ensure we use the result to prevent optimization
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        // Additional coverage: test multiple forward passes (stateful behavior)
        if (offset < Size && (Data[offset++] % 3 == 0)) {
            // Use output as new hidden state for another pass
            try {
                torch::Tensor output2 = gru_cell->forward(input, output);
                if (output2.defined()) {
                    volatile float sum2 = output2.sum().item<float>();
                    (void)sum2;
                }
            }
            catch (...) {
                // Silently ignore errors in secondary pass
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}