#include "fuzzer_utils.h"
#include <iostream>
#include <sstream>

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
        
        // Extract parameters for Linear module first
        int64_t in_features = 1;
        int64_t out_features = 1;
        bool bias = true;
        
        // Get in_features from fuzzer data
        if (offset + sizeof(uint16_t) <= Size) {
            uint16_t raw_in;
            std::memcpy(&raw_in, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            in_features = (raw_in % 64) + 1;  // 1 to 64
        }
        
        // Get out_features from fuzzer data
        if (offset + sizeof(uint16_t) <= Size) {
            uint16_t raw_out;
            std::memcpy(&raw_out, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            out_features = (raw_out % 64) + 1;  // 1 to 64
        }
        
        // Get bias parameter
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Get batch size from fuzzer data
        int64_t batch_size = 1;
        if (offset < Size) {
            batch_size = (Data[offset++] % 16) + 1;  // 1 to 16
        }
        
        // Create the Linear module using LinearOptions
        torch::nn::LinearOptions options(in_features, out_features);
        options.bias(bias);
        torch::nn::Linear linear_module(options);
        
        // Create input tensor with correct shape for the linear layer
        torch::Tensor input_tensor = torch::randn({batch_size, in_features});
        
        // Use remaining fuzzer data to perturb tensor values if available
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t tensor_elements = input_tensor.numel();
            size_t bytes_to_use = std::min(remaining, tensor_elements * sizeof(float));
            
            auto accessor = input_tensor.accessor<float, 2>();
            size_t idx = 0;
            for (int64_t i = 0; i < batch_size && idx < bytes_to_use; i++) {
                for (int64_t j = 0; j < in_features && idx < bytes_to_use; j++) {
                    // Use fuzzer byte to scale the value
                    float scale = static_cast<float>(Data[offset + idx]) / 128.0f - 1.0f;
                    accessor[i][j] *= scale;
                    idx++;
                }
            }
        }
        
        // Apply the Linear module - forward pass
        torch::Tensor output = linear_module->forward(input_tensor);
        
        // Verify output shape
        assert(output.size(0) == batch_size);
        assert(output.size(-1) == out_features);
        
        // Test with requires_grad for backward pass
        torch::Tensor input_grad = torch::randn({batch_size, in_features}, torch::requires_grad());
        torch::Tensor output_grad = linear_module->forward(input_grad);
        output_grad.sum().backward();
        
        // Test parameters access
        auto params = linear_module->parameters();
        assert(params.size() == (bias ? 2 : 1));
        
        // Test named_parameters
        auto named_params = linear_module->named_parameters();
        
        // Test weight and bias access
        torch::Tensor weight = linear_module->weight;
        assert(weight.size(0) == out_features);
        assert(weight.size(1) == in_features);
        
        if (bias) {
            torch::Tensor bias_tensor = linear_module->bias;
            assert(bias_tensor.size(0) == out_features);
        }
        
        // Test serialization/deserialization with actual data
        std::stringstream ss;
        {
            torch::serialize::OutputArchive output_archive;
            linear_module->save(output_archive);
            output_archive.save_to(ss);
        }
        
        {
            torch::nn::Linear loaded_module(options);
            torch::serialize::InputArchive input_archive;
            input_archive.load_from(ss);
            loaded_module->load(input_archive);
            
            // Verify loaded module works
            torch::Tensor loaded_output = loaded_module->forward(input_tensor);
            assert(loaded_output.sizes() == output.sizes());
        }
        
        // Test eval/train modes
        linear_module->eval();
        torch::Tensor eval_output = linear_module->forward(input_tensor);
        
        linear_module->train();
        torch::Tensor train_output = linear_module->forward(input_tensor);
        
        // Test with different input dimensions (2D and 3D inputs)
        if (batch_size > 1) {
            // Test 3D input: (batch, seq_len, in_features)
            int64_t seq_len = (batch_size / 2) + 1;
            torch::Tensor input_3d = torch::randn({batch_size / 2 + 1, seq_len, in_features});
            torch::Tensor output_3d = linear_module->forward(input_3d);
            assert(output_3d.size(-1) == out_features);
        }
        
        // Test zero_grad on module
        linear_module->zero_grad();
        
        // Test to() method for dtype conversion
        try {
            linear_module->to(torch::kFloat64);
            torch::Tensor input_f64 = input_tensor.to(torch::kFloat64);
            torch::Tensor output_f64 = linear_module->forward(input_f64);
        } catch (...) {
            // Silently ignore dtype conversion issues
        }
        
        // Reset to float32
        linear_module->to(torch::kFloat32);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}