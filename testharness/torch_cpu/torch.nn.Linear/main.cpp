#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes for basic parameters
        if (Size < 6) {
            return 0;
        }
        
        // Extract in_features and out_features first
        int64_t in_features = static_cast<int64_t>(Data[offset++]) % 128 + 1;
        int64_t out_features = static_cast<int64_t>(Data[offset++]) % 128 + 1;
        
        // Get bias flag
        bool use_bias = Data[offset++] & 1;
        
        // Get batch size for input tensor
        int64_t batch_size = static_cast<int64_t>(Data[offset++]) % 32 + 1;
        
        // Determine input shape configuration
        uint8_t shape_config = Data[offset++] % 4;
        
        // Create input tensor with correct last dimension
        torch::Tensor input;
        try {
            switch (shape_config) {
                case 0:
                    // 1D input: (in_features,)
                    input = torch::randn({in_features}, torch::kFloat32);
                    break;
                case 1:
                    // 2D input: (batch_size, in_features)
                    input = torch::randn({batch_size, in_features}, torch::kFloat32);
                    break;
                case 2:
                    // 3D input: (batch_size, seq_len, in_features)
                    {
                        int64_t seq_len = static_cast<int64_t>(Data[offset % Size]) % 16 + 1;
                        input = torch::randn({batch_size, seq_len, in_features}, torch::kFloat32);
                    }
                    break;
                default:
                    // 4D input for batched sequence data
                    {
                        int64_t dim1 = static_cast<int64_t>(Data[offset % Size]) % 8 + 1;
                        int64_t dim2 = static_cast<int64_t>(Data[(offset + 1) % Size]) % 8 + 1;
                        input = torch::randn({batch_size, dim1, dim2, in_features}, torch::kFloat32);
                    }
                    break;
            }
        } catch (...) {
            // Silently handle tensor creation failures
            return 0;
        }
        
        // Create Linear layer using LinearOptions
        torch::nn::LinearOptions options(in_features, out_features);
        options.bias(use_bias);
        torch::nn::Linear linear(options);
        
        // Optionally initialize weights with fuzzer data
        if (offset < Size) {
            float weight_scale = static_cast<float>(Data[offset++]) / 255.0f * 2.0f - 1.0f;
            linear->weight.data().fill_(weight_scale);
            
            if (use_bias && offset < Size) {
                float bias_val = static_cast<float>(Data[offset++]) / 255.0f * 2.0f - 1.0f;
                linear->bias.data().fill_(bias_val);
            }
        }
        
        // Apply the linear layer to the input tensor
        torch::Tensor output = linear(input);
        
        // Verify output shape
        auto out_sizes = output.sizes();
        if (out_sizes.size() > 0 && out_sizes.back() != out_features) {
            std::cerr << "Output dimension mismatch" << std::endl;
            return -1;
        }
        
        // Access output elements to ensure computation is performed
        float sum = output.sum().item<float>();
        (void)sum; // Avoid unused variable warning
        
        // Test backward pass with gradient tracking
        if (offset < Size && (Data[offset++] & 1)) {
            // Create a new input with gradient tracking for backward pass
            torch::Tensor grad_input = input.clone().detach().requires_grad_(true);
            torch::Tensor grad_output = linear(grad_input);
            grad_output.backward(torch::ones_like(grad_output));
            
            // Access gradients to ensure they're computed
            if (grad_input.grad().defined()) {
                float grad_sum = grad_input.grad().sum().item<float>();
                (void)grad_sum;
            }
        }
        
        // Test with different input dtypes if we have more data
        if (offset < Size && (Data[offset++] & 1)) {
            try {
                // Test with double precision
                torch::Tensor double_input = input.to(torch::kFloat64);
                torch::nn::Linear double_linear(torch::nn::LinearOptions(in_features, out_features).bias(use_bias));
                double_linear->to(torch::kFloat64);
                torch::Tensor double_output = double_linear(double_input);
                (void)double_output.sum().item<double>();
            } catch (...) {
                // Silently ignore dtype conversion issues
            }
        }
        
        // Test eval/train mode switching
        if (offset < Size) {
            if (Data[offset++] & 1) {
                linear->eval();
                torch::Tensor eval_output = linear(input);
                (void)eval_output;
            } else {
                linear->train();
                torch::Tensor train_output = linear(input);
                (void)train_output;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}