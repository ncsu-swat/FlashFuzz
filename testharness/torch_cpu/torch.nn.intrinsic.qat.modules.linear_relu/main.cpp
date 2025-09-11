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
        
        // Early exit if not enough data
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Linear module
        int64_t in_features = 0;
        int64_t out_features = 0;
        
        // Get in_features from input tensor if possible
        if (input.dim() >= 1) {
            in_features = input.size(-1);
        } else {
            // For scalar tensors, use a small value
            in_features = 4;
        }
        
        // Get out_features from remaining data if available
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is reasonable
            out_features = std::abs(out_features) % 32 + 1;
        } else {
            out_features = 4;
        }
        
        // Create Linear module followed by ReLU
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features));
        torch::nn::ReLU relu;
        
        // Get bias parameter
        bool bias = true;
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
            if (!bias && linear->bias.defined()) {
                linear->bias = torch::Tensor();
            }
        }
        
        // Reshape input tensor if needed to match expected input shape for Linear
        if (input.dim() == 0) {
            // Scalar tensor needs to be reshaped to at least 1D
            input = input.reshape({1, in_features});
        } else if (input.dim() == 1) {
            // 1D tensor needs to be reshaped to 2D for linear layer
            input = input.reshape({1, input.size(0)});
            
            // If the reshaped tensor doesn't match in_features, we need to adjust
            if (input.size(1) != in_features) {
                input = input.reshape({1, in_features});
            }
        } else {
            // For higher dimensions, ensure the last dimension matches in_features
            std::vector<int64_t> new_shape = input.sizes().vec();
            new_shape[new_shape.size() - 1] = in_features;
            input = input.reshape(new_shape);
        }
        
        // Apply the Linear + ReLU modules
        torch::Tensor linear_output = linear->forward(input);
        torch::Tensor output = relu->forward(linear_output);
        
        // Try different training modes
        if (offset < Size) {
            uint8_t mode_op = Data[offset++];
            
            switch (mode_op % 4) {
                case 0:
                    linear->train();
                    relu->train();
                    linear->forward(input);
                    break;
                case 1:
                    linear->eval();
                    relu->eval();
                    linear->forward(input);
                    break;
                case 2:
                    // Access weight parameter
                    if (linear->weight.defined()) {
                        auto weight_sum = linear->weight.sum();
                    }
                    break;
                case 3:
                    // Try different input
                    auto new_input = torch::randn({1, in_features});
                    auto new_linear_out = linear->forward(new_input);
                    auto new_output = relu->forward(new_linear_out);
                    break;
            }
        }
        
        // Try quantization simulation if we have enough data
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            try {
                // Simulate quantization by clamping values
                auto clamped_output = torch::clamp(output, -128.0, 127.0);
                auto rounded_output = torch::round(clamped_output);
            } catch (...) {
                // Ignore quantization simulation errors
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
