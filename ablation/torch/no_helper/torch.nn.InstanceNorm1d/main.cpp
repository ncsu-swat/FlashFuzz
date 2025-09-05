#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from the fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t& offset, size_t size, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 16) return 0;  // Need minimum bytes for basic parameters
    
    size_t offset = 0;
    
    try {
        // Consume parameters for InstanceNorm1d
        int32_t num_features;
        if (!consumeBytes(data, offset, size, num_features)) return 0;
        num_features = std::abs(num_features % 2048) + 1;  // Limit to reasonable range [1, 2048]
        
        float eps;
        if (!consumeBytes(data, offset, size, eps)) return 0;
        eps = std::abs(eps);
        if (eps < 1e-10f) eps = 1e-5f;  // Ensure numerical stability
        
        float momentum;
        if (!consumeBytes(data, offset, size, momentum)) return 0;
        momentum = std::abs(momentum);
        while (momentum > 1.0f) momentum /= 10.0f;  // Clamp to [0, 1]
        
        uint8_t flags;
        if (!consumeBytes(data, offset, size, flags)) return 0;
        bool affine = flags & 0x01;
        bool track_running_stats = flags & 0x02;
        bool use_3d_input = flags & 0x04;
        bool eval_mode = flags & 0x08;
        bool requires_grad = flags & 0x10;
        
        // Create InstanceNorm1d module
        auto options = torch::nn::InstanceNorm1dOptions(num_features)
            .eps(eps)
            .momentum(momentum)
            .affine(affine)
            .track_running_stats(track_running_stats);
        
        torch::nn::InstanceNorm1d norm(options);
        
        if (eval_mode) {
            norm->eval();
        } else {
            norm->train();
        }
        
        // Generate input tensor dimensions
        int32_t batch_size = 1;
        int32_t length = 1;
        
        if (use_3d_input) {
            if (!consumeBytes(data, offset, size, batch_size)) return 0;
            batch_size = std::abs(batch_size % 128) + 1;  // [1, 128]
        }
        
        if (!consumeBytes(data, offset, size, length)) return 0;
        length = std::abs(length % 256) + 1;  // [1, 256]
        
        // Create input tensor
        torch::Tensor input;
        if (use_3d_input) {
            input = torch::randn({batch_size, num_features, length}, 
                                torch::TensorOptions().requires_grad(requires_grad));
        } else {
            input = torch::randn({num_features, length}, 
                                torch::TensorOptions().requires_grad(requires_grad));
        }
        
        // Fill input with fuzzed data if enough bytes remain
        size_t tensor_bytes = input.numel() * sizeof(float);
        if (offset + tensor_bytes <= size) {
            float* input_data = input.data_ptr<float>();
            std::memcpy(input_data, data + offset, tensor_bytes);
            offset += tensor_bytes;
            
            // Sanitize input to avoid NaN/Inf propagation
            for (int i = 0; i < input.numel(); ++i) {
                if (!std::isfinite(input_data[i])) {
                    input_data[i] = 0.0f;
                }
            }
        }
        
        // Test forward pass
        torch::Tensor output = norm->forward(input);
        
        // Verify output shape matches input
        if (output.sizes() != input.sizes()) {
            std::cerr << "Shape mismatch!" << std::endl;
            return -1;
        }
        
        // Test backward pass if gradients are enabled
        if (requires_grad && !eval_mode) {
            torch::Tensor loss = output.sum();
            loss.backward();
            
            // Check gradients exist
            if (input.grad().defined()) {
                torch::Tensor grad = input.grad();
                // Force computation
                grad.sum().item<float>();
            }
        }
        
        // Test edge cases with different input variations
        if (offset < size) {
            uint8_t edge_case = data[offset++];
            
            if (edge_case & 0x01) {
                // Test with zeros
                torch::Tensor zero_input = torch::zeros_like(input);
                torch::Tensor zero_output = norm->forward(zero_input);
                zero_output.sum().item<float>();
            }
            
            if (edge_case & 0x02) {
                // Test with ones
                torch::Tensor ones_input = torch::ones_like(input);
                torch::Tensor ones_output = norm->forward(ones_input);
                ones_output.sum().item<float>();
            }
            
            if (edge_case & 0x04) {
                // Test with very small values
                torch::Tensor small_input = input * 1e-8f;
                torch::Tensor small_output = norm->forward(small_input);
                small_output.sum().item<float>();
            }
            
            if (edge_case & 0x08) {
                // Test with very large values
                torch::Tensor large_input = input * 1e8f;
                torch::Tensor large_output = norm->forward(large_input);
                large_output.sum().item<float>();
            }
        }
        
        // Test state persistence for track_running_stats
        if (track_running_stats && !eval_mode) {
            // Multiple forward passes to update running stats
            for (int i = 0; i < 3; ++i) {
                torch::Tensor iter_input = torch::randn_like(input);
                norm->forward(iter_input);
            }
        }
        
        // Test module serialization/deserialization
        if (offset < size && data[offset] & 0x01) {
            std::stringstream stream;
            torch::save(norm, stream);
            torch::nn::InstanceNorm1d loaded_norm(options);
            torch::load(loaded_norm, stream);
            torch::Tensor loaded_output = loaded_norm->forward(input);
            loaded_output.sum().item<float>();
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for edge cases
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}