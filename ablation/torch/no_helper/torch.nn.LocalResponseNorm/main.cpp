#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeValue(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 16) return 0;  // Need minimum bytes for basic parameters
    
    try {
        size_t offset = 0;
        
        // Consume LocalResponseNorm parameters
        int64_t norm_size;
        double alpha, beta, k;
        
        if (!consumeValue(data, size, offset, norm_size)) return 0;
        if (!consumeValue(data, size, offset, alpha)) return 0;
        if (!consumeValue(data, size, offset, beta)) return 0;
        if (!consumeValue(data, size, offset, k)) return 0;
        
        // Constrain norm_size to reasonable range
        norm_size = (std::abs(norm_size) % 100) + 1;  // [1, 100]
        
        // Create LocalResponseNorm module
        torch::nn::LocalResponseNorm lrn_module(torch::nn::LocalResponseNormOptions(norm_size)
            .alpha(alpha)
            .beta(beta)
            .k(k));
        
        // Consume tensor configuration
        uint8_t num_dims;
        if (!consumeValue(data, size, offset, num_dims)) return 0;
        num_dims = (num_dims % 5) + 2;  // [2, 6] dimensions (N, C, ...)
        
        // Build tensor shape
        std::vector<int64_t> shape;
        for (int i = 0; i < num_dims; ++i) {
            uint8_t dim_size;
            if (!consumeValue(data, size, offset, dim_size)) {
                dim_size = 1 + (i % 10);  // Default fallback
            }
            // First dim is batch, second is channels, rest are spatial
            if (i == 0) {
                shape.push_back((dim_size % 16) + 1);  // Batch: [1, 16]
            } else if (i == 1) {
                shape.push_back((dim_size % 32) + 1);  // Channels: [1, 32]
            } else {
                shape.push_back((dim_size % 24) + 1);  // Spatial: [1, 24]
            }
        }
        
        // Determine dtype
        uint8_t dtype_selector;
        if (!consumeValue(data, size, offset, dtype_selector)) {
            dtype_selector = 0;
        }
        
        torch::Tensor input;
        switch (dtype_selector % 3) {
            case 0:
                input = torch::randn(shape, torch::kFloat32);
                break;
            case 1:
                input = torch::randn(shape, torch::kFloat64);
                break;
            case 2:
                input = torch::randn(shape, torch::kFloat16);
                break;
        }
        
        // Add some variety to input values
        uint8_t value_modifier;
        if (consumeValue(data, size, offset, value_modifier)) {
            switch (value_modifier % 5) {
                case 0:
                    input = torch::zeros_like(input);
                    break;
                case 1:
                    input = torch::ones_like(input);
                    break;
                case 2:
                    input = input * 1000.0;  // Large values
                    break;
                case 3:
                    input = input * 0.001;   // Small values
                    break;
                case 4:
                    // Add some NaN/Inf
                    if (input.numel() > 0) {
                        input.view(-1)[0] = std::numeric_limits<float>::quiet_NaN();
                        if (input.numel() > 1) {
                            input.view(-1)[1] = std::numeric_limits<float>::infinity();
                        }
                    }
                    break;
            }
        }
        
        // Test with different memory layouts
        uint8_t layout_modifier;
        if (consumeValue(data, size, offset, layout_modifier)) {
            switch (layout_modifier % 3) {
                case 0:
                    // Contiguous (default)
                    break;
                case 1:
                    // Non-contiguous via transpose
                    if (shape.size() >= 3) {
                        input = input.transpose(0, 2).contiguous().transpose(0, 2);
                    }
                    break;
                case 2:
                    // Non-contiguous via slice
                    if (input.size(0) > 1) {
                        input = input.slice(0, 0, input.size(0), 2);
                    }
                    break;
            }
        }
        
        // Apply LocalResponseNorm
        torch::Tensor output = lrn_module->forward(input);
        
        // Perform some operations to ensure computation happens
        if (output.numel() > 0) {
            auto sum = output.sum();
            auto mean = output.mean();
            
            // Test backward pass
            uint8_t test_backward;
            if (consumeValue(data, size, offset, test_backward) && (test_backward % 2 == 0)) {
                if (input.requires_grad()) {
                    input.set_requires_grad(true);
                    output = lrn_module->forward(input);
                    auto loss = output.sum();
                    loss.backward();
                }
            }
        }
        
        // Test edge cases with different input configurations
        if (offset < size) {
            // Test with batch size 0 (should handle gracefully or throw)
            try {
                std::vector<int64_t> zero_shape = shape;
                zero_shape[0] = 0;
                torch::Tensor zero_input = torch::empty(zero_shape);
                torch::Tensor zero_output = lrn_module->forward(zero_input);
            } catch (...) {
                // Expected for invalid shapes
            }
            
            // Test with single channel
            try {
                std::vector<int64_t> single_channel_shape = shape;
                single_channel_shape[1] = 1;
                torch::Tensor single_input = torch::randn(single_channel_shape);
                torch::Tensor single_output = lrn_module->forward(single_input);
            } catch (...) {
                // Handle gracefully
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid inputs
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Unknown exception
        return -1;
    }
    
    return 0;
}