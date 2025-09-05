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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    try {
        if (size < 4) return 0;
        
        size_t offset = 0;
        
        // Consume configuration bytes
        uint8_t num_dims = 0;
        uint8_t dtype_selector = 0;
        uint8_t requires_grad = 0;
        uint8_t use_cuda = 0;
        
        if (!consumeBytes(data, offset, size, num_dims)) return 0;
        if (!consumeBytes(data, offset, size, dtype_selector)) return 0;
        if (!consumeBytes(data, offset, size, requires_grad)) return 0;
        if (!consumeBytes(data, offset, size, use_cuda)) return 0;
        
        // Limit dimensions to reasonable range
        num_dims = num_dims % 8;
        
        // Build shape vector
        std::vector<int64_t> shape;
        for (int i = 0; i < num_dims; ++i) {
            uint8_t dim_size = 0;
            if (!consumeBytes(data, offset, size, dim_size)) {
                shape.push_back(1);
            } else {
                // Allow 0-sized dimensions and various sizes
                shape.push_back(dim_size % 16);
            }
        }
        
        // Select dtype
        torch::ScalarType dtype;
        switch (dtype_selector % 4) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            case 3: dtype = torch::kBFloat16; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Determine device
        torch::Device device(torch::kCPU);
        if (use_cuda % 2 == 1 && torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
        }
        
        // Create options
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .requires_grad((requires_grad % 2) == 1);
        
        // Create input tensor
        torch::Tensor input;
        
        // Handle empty shape case
        if (shape.empty()) {
            // Scalar tensor
            float scalar_val = 0.0f;
            consumeBytes(data, offset, size, scalar_val);
            input = torch::tensor(scalar_val, options);
        } else {
            // Multi-dimensional tensor
            int64_t total_elements = 1;
            for (auto dim : shape) {
                total_elements *= dim;
            }
            
            if (total_elements == 0) {
                // Empty tensor
                input = torch::empty(shape, options);
            } else if (total_elements > 10000) {
                // Limit total elements to prevent OOM
                input = torch::randn(shape, options);
            } else {
                // Fill with fuzzed data
                std::vector<float> values;
                values.reserve(total_elements);
                
                for (int64_t i = 0; i < total_elements; ++i) {
                    float val = 0.0f;
                    if (offset < size) {
                        consumeBytes(data, offset, size, val);
                        // Handle special values
                        if (std::isnan(val) || std::isinf(val)) {
                            values.push_back(val);
                        } else {
                            values.push_back(val);
                        }
                    } else {
                        values.push_back(static_cast<float>(i) / total_elements);
                    }
                }
                
                auto cpu_tensor = torch::from_blob(values.data(), shape, torch::kFloat32).clone();
                input = cpu_tensor.to(options);
            }
        }
        
        // Create and apply Tanhshrink
        torch::nn::Tanhshrink tanhshrink;
        
        // Test forward pass
        torch::Tensor output = tanhshrink->forward(input);
        
        // Verify output shape matches input
        if (output.sizes() != input.sizes()) {
            std::cerr << "Shape mismatch!" << std::endl;
        }
        
        // Test with different input variations
        if (offset < size) {
            uint8_t variation = 0;
            consumeBytes(data, offset, size, variation);
            
            switch (variation % 5) {
                case 0:
                    // Test with negative values
                    if (input.numel() > 0) {
                        torch::Tensor neg_input = -input;
                        torch::Tensor neg_output = tanhshrink->forward(neg_input);
                    }
                    break;
                case 1:
                    // Test with zeros
                    if (input.numel() > 0) {
                        torch::Tensor zero_input = torch::zeros_like(input);
                        torch::Tensor zero_output = tanhshrink->forward(zero_input);
                    }
                    break;
                case 2:
                    // Test with ones
                    if (input.numel() > 0) {
                        torch::Tensor ones_input = torch::ones_like(input);
                        torch::Tensor ones_output = tanhshrink->forward(ones_input);
                    }
                    break;
                case 3:
                    // Test with large values
                    if (input.numel() > 0) {
                        torch::Tensor large_input = input * 1000.0;
                        torch::Tensor large_output = tanhshrink->forward(large_input);
                    }
                    break;
                case 4:
                    // Test with small values
                    if (input.numel() > 0) {
                        torch::Tensor small_input = input * 0.001;
                        torch::Tensor small_output = tanhshrink->forward(small_input);
                    }
                    break;
            }
        }
        
        // Test gradient computation if applicable
        if (input.requires_grad() && input.numel() > 0) {
            try {
                output.sum().backward();
            } catch (...) {
                // Gradient computation might fail for some dtypes
            }
        }
        
        // Test in-place operation
        if (input.numel() > 0 && offset < size) {
            uint8_t do_inplace = 0;
            consumeBytes(data, offset, size, do_inplace);
            if (do_inplace % 2 == 1) {
                torch::Tensor input_copy = input.clone();
                tanhshrink->forward(input_copy);
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors
        return 0;
    } catch (const std::bad_alloc& e) {
        // Memory allocation failures
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Unknown exceptions
        return 0;
    }
    
    return 0;
}