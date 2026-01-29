#include "fuzzer_utils.h"
#include <iostream>

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
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Parse input_size (1-64)
        int64_t input_size = 1;
        if (offset < Size) {
            input_size = (Data[offset++] % 64) + 1;
        }

        // Parse hidden_size (1-64)
        int64_t hidden_size = 1;
        if (offset < Size) {
            hidden_size = (Data[offset++] % 64) + 1;
        }

        // Parse batch_size (1-16)
        int64_t batch_size = 1;
        if (offset < Size) {
            batch_size = (Data[offset++] % 16) + 1;
        }

        // Parse bias flag
        bool bias = true;
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }

        // Parse nonlinearity type
        bool use_tanh = true;
        if (offset < Size) {
            use_tanh = Data[offset++] & 0x1;
        }

        // Create RNNCell with parsed options
        torch::nn::RNNCellOptions options(input_size, hidden_size);
        options.bias(bias);
        
        // Set nonlinearity based on parsed value
        if (use_tanh) {
            options.nonlinearity(torch::kTanh);
        } else {
            options.nonlinearity(torch::kReLU);
        }

        torch::nn::RNNCell cell(options);

        // Create input tensor with correct dimensions: (batch_size, input_size)
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
            // Calculate total elements and reshape safely
            int64_t numel = input.numel();
            if (numel >= batch_size * input_size) {
                input = input.flatten().slice(0, 0, batch_size * input_size)
                            .reshape({batch_size, input_size});
            } else if (numel > 0) {
                // Pad with zeros if not enough elements
                auto flat = input.flatten();
                input = torch::zeros({batch_size, input_size}, input.options());
                input.flatten().slice(0, 0, numel).copy_(flat);
            } else {
                input = torch::randn({batch_size, input_size});
            }
        } else {
            input = torch::randn({batch_size, input_size});
        }

        // Create hidden state tensor with correct dimensions: (batch_size, hidden_size)
        torch::Tensor hx;
        if (offset < Size) {
            hx = fuzzer_utils::createTensor(Data, Size, offset);
            int64_t numel = hx.numel();
            if (numel >= batch_size * hidden_size) {
                hx = hx.flatten().slice(0, 0, batch_size * hidden_size)
                       .reshape({batch_size, hidden_size});
            } else if (numel > 0) {
                auto flat = hx.flatten();
                hx = torch::zeros({batch_size, hidden_size}, hx.options());
                hx.flatten().slice(0, 0, numel).copy_(flat);
            } else {
                hx = torch::zeros({batch_size, hidden_size});
            }
        } else {
            hx = torch::zeros({batch_size, hidden_size});
        }

        // Ensure tensors are float type (RNNCell requires float)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat);
        }
        if (!hx.is_floating_point()) {
            hx = hx.to(torch::kFloat);
        }

        // Test 1: Forward pass with hidden state
        torch::Tensor output = cell->forward(input, hx);

        // Verify output shape
        if (output.size(0) != batch_size || output.size(1) != hidden_size) {
            std::cerr << "Unexpected output shape" << std::endl;
        }

        // Test 2: Forward pass without hidden state (uses zeros)
        torch::Tensor output2 = cell->forward(input);

        // Test 3: Chain multiple steps to test recurrent behavior
        torch::Tensor h = hx;
        for (int step = 0; step < 3; step++) {
            h = cell->forward(input, h);
        }

        // Use results to prevent optimization
        auto sum = output.sum() + output2.sum() + h.sum();
        if (sum.item<float>() == std::numeric_limits<float>::quiet_NaN()) {
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}