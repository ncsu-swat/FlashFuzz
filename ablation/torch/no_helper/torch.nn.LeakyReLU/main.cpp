#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from the fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) {
        return false;
    }
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) {
        return 0;
    }

    try {
        size_t offset = 0;
        
        // Consume configuration bytes
        uint8_t config_byte = 0;
        if (!consumeBytes(data, size, offset, config_byte)) {
            return 0;
        }
        
        // Extract configuration from config_byte
        bool inplace = (config_byte & 0x01) != 0;
        bool use_cuda = (config_byte & 0x02) != 0 && torch::cuda::is_available();
        uint8_t dtype_selector = (config_byte >> 2) & 0x07;
        uint8_t ndims = (config_byte >> 5) & 0x07; // 0-7 dimensions
        
        // Consume negative_slope parameter
        float negative_slope_raw = 0.0f;
        if (!consumeBytes(data, size, offset, negative_slope_raw)) {
            return 0;
        }
        // Map to reasonable range for negative_slope
        double negative_slope = static_cast<double>(std::fabs(negative_slope_raw));
        if (std::isnan(negative_slope) || std::isinf(negative_slope)) {
            negative_slope = 0.01;
        }
        negative_slope = std::fmod(negative_slope, 10.0); // Keep it reasonable
        
        // Determine tensor dtype
        torch::ScalarType dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            case 3: dtype = torch::kBFloat16; break;
            case 4: dtype = torch::kInt32; break;
            case 5: dtype = torch::kInt64; break;
            case 6: dtype = torch::kInt8; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Build tensor shape
        std::vector<int64_t> shape;
        for (uint8_t i = 0; i < ndims; ++i) {
            uint8_t dim_size = 0;
            if (!consumeBytes(data, size, offset, dim_size)) {
                break;
            }
            // Allow 0-sized dimensions for edge cases
            shape.push_back(static_cast<int64_t>(dim_size % 32)); // Limit dimension size
        }
        
        // Create input tensor
        torch::Tensor input;
        if (shape.empty()) {
            // Scalar tensor
            input = torch::randn({}, torch::TensorOptions().dtype(dtype));
        } else {
            // Check if total elements is reasonable
            int64_t total_elements = 1;
            for (auto dim : shape) {
                if (dim > 0) {
                    if (total_elements > 1000000 / dim) {
                        // Prevent excessive memory usage
                        return 0;
                    }
                    total_elements *= dim;
                }
            }
            
            input = torch::randn(shape, torch::TensorOptions().dtype(dtype));
        }
        
        // Fill tensor with fuzzed data if enough bytes remain
        if (offset < size) {
            size_t remaining = size - offset;
            size_t tensor_bytes = input.numel() * input.element_size();
            
            if (tensor_bytes > 0 && remaining > 0) {
                size_t copy_bytes = std::min(remaining, tensor_bytes);
                if (input.is_contiguous()) {
                    std::memcpy(input.data_ptr(), data + offset, copy_bytes);
                }
            }
        }
        
        // Move to CUDA if requested and available
        if (use_cuda) {
            input = input.cuda();
        }
        
        // Create LeakyReLU module
        torch::nn::LeakyReLU leaky_relu_module(torch::nn::LeakyReLUOptions()
            .negative_slope(negative_slope)
            .inplace(inplace));
        
        // Apply LeakyReLU
        torch::Tensor output;
        if (inplace && input.dtype().isFloatingPoint()) {
            // For inplace operation, clone first to avoid modifying original
            torch::Tensor input_clone = input.clone();
            output = leaky_relu_module->forward(input_clone);
            
            // Verify inplace behavior
            if (!output.is_same(input_clone)) {
                // This shouldn't happen for inplace=true
            }
        } else {
            output = leaky_relu_module->forward(input);
        }
        
        // Also test functional API
        torch::Tensor func_output = torch::nn::functional::leaky_relu(
            input, 
            torch::nn::functional::LeakyReLUFuncOptions()
                .negative_slope(negative_slope)
                .inplace(false)
        );
        
        // Test with different edge cases
        if (input.numel() > 0) {
            // Test with all positive values
            torch::Tensor positive_input = torch::abs(input);
            torch::Tensor pos_output = torch::nn::functional::leaky_relu(
                positive_input,
                torch::nn::functional::LeakyReLUFuncOptions()
                    .negative_slope(negative_slope)
            );
            
            // Test with all negative values
            torch::Tensor negative_input = -torch::abs(input);
            torch::Tensor neg_output = torch::nn::functional::leaky_relu(
                negative_input,
                torch::nn::functional::LeakyReLUFuncOptions()
                    .negative_slope(negative_slope)
            );
            
            // Test with zeros
            torch::Tensor zero_input = torch::zeros_like(input);
            torch::Tensor zero_output = torch::nn::functional::leaky_relu(
                zero_input,
                torch::nn::functional::LeakyReLUFuncOptions()
                    .negative_slope(negative_slope)
            );
        }
        
        // Test gradient computation if tensor requires grad
        if (input.dtype().isFloatingPoint() && input.numel() > 0 && input.numel() < 10000) {
            torch::Tensor grad_input = input.clone().requires_grad_(true);
            torch::Tensor grad_output = torch::nn::functional::leaky_relu(
                grad_input,
                torch::nn::functional::LeakyReLUFuncOptions()
                    .negative_slope(negative_slope)
            );
            
            if (grad_output.numel() > 0) {
                torch::Tensor loss = grad_output.sum();
                loss.backward();
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions
        return 0;
    }
    
    return 0;
}