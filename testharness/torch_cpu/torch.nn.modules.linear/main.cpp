#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        // Need at least a few bytes for parameters
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract in_features from data (1-64 range)
        int64_t in_features = (Data[offset++] % 64) + 1;
        
        // Extract out_features from data (1-64 range)
        int64_t out_features = (Data[offset++] % 64) + 1;
        
        // Extract bias flag
        bool bias = Data[offset++] & 0x1;
        
        // Extract batch size (1-16 range)
        int64_t batch_size = (Data[offset++] % 16) + 1;
        
        // Create the linear module
        torch::nn::LinearOptions options(in_features, out_features);
        options.bias(bias);
        torch::nn::Linear linear_module(options);
        
        // Create input tensor with correct shape for the linear layer
        torch::Tensor input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
        
        // Reshape or create a properly shaped input tensor
        torch::Tensor shaped_input;
        try {
            // Create a tensor with the correct last dimension
            shaped_input = torch::randn({batch_size, in_features});
            
            // If we have fuzzer-generated data, try to use it to fill values
            if (input.numel() > 0) {
                auto flat_input = input.flatten();
                int64_t copy_size = std::min(flat_input.numel(), shaped_input.numel());
                if (copy_size > 0 && flat_input.scalar_type() == torch::kFloat) {
                    shaped_input.flatten().slice(0, 0, copy_size).copy_(
                        flat_input.slice(0, 0, copy_size));
                }
            }
        } catch (...) {
            // If shaping fails, create a simple valid input
            shaped_input = torch::randn({batch_size, in_features});
        }
        
        // Test 1: Basic forward pass
        try {
            torch::Tensor output = linear_module->forward(shaped_input);
        } catch (const c10::Error& e) {
            // Expected exceptions are fine
        }
        
        // Test 2: Forward pass with 1D input (single sample)
        try {
            torch::Tensor input_1d = torch::randn({in_features});
            torch::Tensor output = linear_module->forward(input_1d);
        } catch (const c10::Error& e) {
            // Expected exceptions are fine
        }
        
        // Test 3: Forward pass with 3D input (sequence data)
        try {
            int64_t seq_len = (batch_size % 8) + 1;
            torch::Tensor input_3d = torch::randn({batch_size, seq_len, in_features});
            torch::Tensor output = linear_module->forward(input_3d);
        } catch (const c10::Error& e) {
            // Expected exceptions are fine
        }
        
        // Test 4: Test with different dtypes
        try {
            torch::Tensor double_input = shaped_input.to(torch::kDouble);
            auto double_module = torch::nn::Linear(
                torch::nn::LinearOptions(in_features, out_features).bias(bias));
            double_module->to(torch::kDouble);
            torch::Tensor output = double_module->forward(double_input);
        } catch (const c10::Error& e) {
            // Expected exceptions are fine
        }
        
        // Test 5: Test with zero weights
        try {
            linear_module->weight.zero_();
            if (bias && linear_module->bias.defined()) {
                linear_module->bias.zero_();
            }
            torch::Tensor output = linear_module->forward(shaped_input);
        } catch (const c10::Error& e) {
            // Expected exceptions are fine
        }
        
        // Test 6: Test module parameters iteration
        try {
            for (auto& param : linear_module->parameters()) {
                auto grad = torch::ones_like(param);
            }
        } catch (const c10::Error& e) {
            // Expected exceptions are fine
        }
        
        // Test 7: Test named_parameters
        try {
            for (auto& named_param : linear_module->named_parameters()) {
                auto name = named_param.key();
                auto param = named_param.value();
            }
        } catch (const c10::Error& e) {
            // Expected exceptions are fine
        }
        
        // Test 8: Test clone
        try {
            auto cloned_module = std::dynamic_pointer_cast<torch::nn::LinearImpl>(
                linear_module->clone());
            if (cloned_module) {
                torch::Tensor output = cloned_module->forward(shaped_input);
            }
        } catch (const c10::Error& e) {
            // Expected exceptions are fine
        }
        
        // Test 9: Test eval and train modes
        try {
            linear_module->eval();
            torch::Tensor output_eval = linear_module->forward(shaped_input);
            linear_module->train();
            torch::Tensor output_train = linear_module->forward(shaped_input);
        } catch (const c10::Error& e) {
            // Expected exceptions are fine
        }
        
        // Test 10: Test with extreme values in input
        try {
            torch::Tensor extreme_input = torch::full({batch_size, in_features}, 1e10);
            torch::Tensor output = linear_module->forward(extreme_input);
        } catch (const c10::Error& e) {
            // Expected exceptions are fine
        }
        
        // Test 11: Test with NaN/Inf input
        try {
            torch::Tensor nan_input = torch::full({batch_size, in_features}, 
                                                   std::numeric_limits<float>::quiet_NaN());
            torch::Tensor output = linear_module->forward(nan_input);
        } catch (const c10::Error& e) {
            // Expected exceptions are fine
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}