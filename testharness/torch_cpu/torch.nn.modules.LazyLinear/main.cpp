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
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - Linear requires at least 1D input
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
            // Linear needs at least 1D input
            if (input.dim() < 1) {
                return 0;
            }
            // Ensure input is float type for Linear operations
            if (!input.is_floating_point()) {
                input = input.to(torch::kFloat32);
            }
        } catch (const std::exception& e) {
            return 0;
        }
        
        // Get in_features from input tensor's last dimension
        int64_t in_features = input.size(-1);
        if (in_features <= 0) {
            return 0;
        }
        
        // Get out_features from the data
        int64_t out_features = 1;
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_out_features;
            std::memcpy(&raw_out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure out_features is positive and reasonable
            out_features = std::abs(raw_out_features) % 512 + 1;
        } else if (offset < Size) {
            out_features = (Data[offset++] % 64) + 1;
        }
        
        // Get bias parameter if available
        bool use_bias = true;
        if (offset < Size) {
            use_bias = Data[offset++] & 0x1;
        }
        
        // Create Linear module with explicit in_features (simulating what LazyLinear does after materialization)
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features).bias(use_bias));
        
        // Apply the module to the input tensor
        torch::Tensor output;
        try {
            output = linear->forward(input);
        } catch (const c10::Error& e) {
            // Shape mismatches or invalid operations are expected
            return 0;
        }
        
        // Access weight to ensure it's valid
        if (linear->weight.defined()) {
            auto weight = linear->weight;
            (void)weight.sizes();
            
            // If bias is enabled, access it
            if (use_bias && linear->bias.defined()) {
                auto bias_tensor = linear->bias;
                (void)bias_tensor.sizes();
            }
        }
        
        // Test with different inputs (same in_features dimension)
        if (offset < Size && (Data[offset] & 0x1)) {
            offset++;
            try {
                // Create another input with same last dimension
                std::vector<int64_t> new_sizes;
                if (offset < Size) {
                    int batch = (Data[offset++] % 8) + 1;
                    new_sizes.push_back(batch);
                }
                new_sizes.push_back(in_features);  // Must match in_features
                
                torch::Tensor input2 = torch::randn(new_sizes);
                torch::Tensor output2 = linear->forward(input2);
                (void)output2;
            } catch (const c10::Error& e) {
                // Expected for shape mismatches
            }
        }
        
        // Test backward pass if gradients are tracked
        if (offset < Size && (Data[offset] & 0x1)) {
            offset++;
            try {
                torch::Tensor grad_input = input.clone().detach().requires_grad_(true);
                torch::nn::Linear linear2(torch::nn::LinearOptions(in_features, out_features).bias(use_bias));
                torch::Tensor grad_output = linear2->forward(grad_input);
                grad_output.sum().backward();
                
                // Access gradients
                if (grad_input.grad().defined()) {
                    (void)grad_input.grad().sizes();
                }
            } catch (const c10::Error& e) {
                // Gradient computation may fail for certain inputs
            }
        }
        
        // Test eval mode
        linear->eval();
        try {
            torch::Tensor eval_output = linear->forward(input);
            (void)eval_output;
        } catch (const c10::Error& e) {
            // May fail for certain inputs
        }
        
        // Test train mode
        linear->train();
        try {
            torch::Tensor train_output = linear->forward(input);
            (void)train_output;
        } catch (const c10::Error& e) {
            // May fail for certain inputs
        }
        
        // Test parameters() iteration
        try {
            for (const auto& param : linear->parameters()) {
                (void)param.sizes();
            }
        } catch (const c10::Error& e) {
            // May fail
        }
        
        // Test named_parameters() iteration
        try {
            for (const auto& named_param : linear->named_parameters()) {
                (void)named_param.key();
                (void)named_param.value().sizes();
            }
        } catch (const c10::Error& e) {
            // May fail
        }
        
        // Test zero_grad
        try {
            linear->zero_grad();
        } catch (const c10::Error& e) {
            // May fail
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}