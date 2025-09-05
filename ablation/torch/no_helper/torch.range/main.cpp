#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t& offset, size_t size, T& value) {
    if (offset + sizeof(T) > size) {
        return false;
    }
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 16) {
        return 0;  // Need minimum bytes for parameters
    }

    size_t offset = 0;
    
    try {
        // Consume parameters for torch.range
        float start, end, step;
        uint8_t dtype_idx, device_idx, requires_grad, use_out;
        
        if (!consumeBytes(data, offset, size, start)) return 0;
        if (!consumeBytes(data, offset, size, end)) return 0;
        if (!consumeBytes(data, offset, size, step)) return 0;
        if (!consumeBytes(data, offset, size, dtype_idx)) return 0;
        if (!consumeBytes(data, offset, size, device_idx)) return 0;
        if (!consumeBytes(data, offset, size, requires_grad)) return 0;
        if (!consumeBytes(data, offset, size, use_out)) return 0;
        
        // Sanitize step to avoid infinite loops
        if (std::abs(step) < 1e-6f) {
            step = (step >= 0) ? 1e-6f : -1e-6f;
        }
        
        // Limit range size to prevent OOM
        float range_size = std::abs((end - start) / step);
        if (range_size > 1000000) {
            // Scale down the range
            end = start + step * 1000000;
        }
        
        // Map dtype index to actual dtype
        std::vector<torch::ScalarType> dtypes = {
            torch::kFloat32,
            torch::kFloat64,
            torch::kInt32,
            torch::kInt64,
            torch::kInt16,
            torch::kInt8,
            torch::kUInt8,
            torch::kBool,
            torch::kHalf,
            torch::kBFloat16
        };
        torch::ScalarType dtype = dtypes[dtype_idx % dtypes.size()];
        
        // Map device index
        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available() && (device_idx % 2 == 1)) {
            device = torch::Device(torch::kCUDA);
        }
        
        // Create options
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .requires_grad(requires_grad & 1);
        
        torch::Tensor result;
        
        // Test with or without pre-allocated output tensor
        if (use_out & 1) {
            // Calculate expected size for output tensor
            int64_t expected_size = static_cast<int64_t>(std::floor((end - start) / step)) + 1;
            if (expected_size <= 0) expected_size = 1;
            
            // Create output tensor with matching size
            torch::Tensor out_tensor = torch::empty({expected_size}, options);
            
            // Call torch.range with output tensor
            result = torch::range(start, end, step, out_tensor);
            
            // Verify output tensor was used
            if (result.data_ptr() != out_tensor.data_ptr()) {
                // This shouldn't happen but let's continue
            }
        } else {
            // Call torch.range without output tensor
            result = torch::range(start, end, step, options);
        }
        
        // Exercise the result tensor to trigger any latent issues
        if (result.numel() > 0) {
            // Access some properties
            auto shape = result.sizes();
            auto strides = result.strides();
            auto numel = result.numel();
            auto is_contiguous = result.is_contiguous();
            
            // Try some operations if tensor is not too large
            if (numel < 10000) {
                // Sum operation
                auto sum_val = result.sum();
                
                // Min/max if applicable
                if (dtype != torch::kBool) {
                    auto min_val = result.min();
                    auto max_val = result.max();
                }
                
                // Mean for floating point types
                if (torch::isFloatingType(dtype)) {
                    auto mean_val = result.mean();
                }
                
                // Clone to test memory operations
                auto cloned = result.clone();
                
                // Test gradient operations if requires_grad
                if (requires_grad & 1 && torch::isFloatingType(dtype)) {
                    try {
                        auto grad_result = result * 2.0;
                        if (grad_result.requires_grad()) {
                            grad_result.sum().backward();
                        }
                    } catch (...) {
                        // Gradient operations might fail for some dtypes
                    }
                }
            }
        }
        
        // Test edge cases with special values
        if (offset < size) {
            uint8_t special_case = data[offset] % 5;
            switch (special_case) {
                case 0:
                    // Test with NaN/Inf values for floating types
                    if (torch::isFloatingType(dtype)) {
                        torch::range(NAN, end, step, options);
                        torch::range(start, INFINITY, step, options);
                        torch::range(start, end, NAN, options);
                    }
                    break;
                case 1:
                    // Test with negative step
                    torch::range(end, start, -std::abs(step), options);
                    break;
                case 2:
                    // Test with zero range
                    torch::range(start, start, step, options);
                    break;
                case 3:
                    // Test with very small step
                    torch::range(start, start + 1, 0.00001f, options);
                    break;
                case 4:
                    // Test with large values
                    torch::range(1e6f, 1e6f + 100, 1.0f, options);
                    break;
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