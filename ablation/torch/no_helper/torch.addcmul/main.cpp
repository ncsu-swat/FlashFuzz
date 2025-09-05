#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t& offset, size_t size, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 16) return 0;  // Need minimum bytes for basic config
    
    size_t offset = 0;
    
    try {
        // Consume configuration bytes
        uint8_t dtype_choice = 0;
        uint8_t rank = 0;
        uint8_t use_out = 0;
        uint8_t value_type = 0;
        float value_float = 1.0f;
        int64_t value_int = 1;
        
        if (!consumeBytes(data, offset, size, dtype_choice)) return 0;
        if (!consumeBytes(data, offset, size, rank)) return 0;
        if (!consumeBytes(data, offset, size, use_out)) return 0;
        if (!consumeBytes(data, offset, size, value_type)) return 0;
        if (!consumeBytes(data, offset, size, value_float)) return 0;
        if (!consumeBytes(data, offset, size, value_int)) return 0;
        
        // Limit rank to reasonable value
        rank = (rank % 5) + 1;  // 1-5 dimensions
        
        // Select dtype
        torch::ScalarType dtype;
        switch (dtype_choice % 6) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kInt8; break;
            case 5: dtype = torch::kUInt8; break;
            default: dtype = torch::kFloat32;
        }
        
        // Generate shapes for three tensors (potentially broadcastable)
        std::vector<int64_t> shape_input, shape_tensor1, shape_tensor2;
        for (int i = 0; i < rank; i++) {
            uint8_t dim_input = 1, dim_tensor1 = 1, dim_tensor2 = 1;
            if (consumeBytes(data, offset, size, dim_input))
                shape_input.push_back((dim_input % 10) + 1);  // 1-10
            else
                shape_input.push_back(1);
                
            if (consumeBytes(data, offset, size, dim_tensor1))
                shape_tensor1.push_back((dim_tensor1 % 10) + 1);
            else
                shape_tensor1.push_back(1);
                
            if (consumeBytes(data, offset, size, dim_tensor2))
                shape_tensor2.push_back((dim_tensor2 % 10) + 1);
            else
                shape_tensor2.push_back(1);
        }
        
        // Create tensors with random data
        auto options = torch::TensorOptions().dtype(dtype);
        torch::Tensor input, tensor1, tensor2;
        
        // Initialize tensors based on dtype
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            input = torch::randn(shape_input, options);
            tensor1 = torch::randn(shape_tensor1, options);
            tensor2 = torch::randn(shape_tensor2, options);
        } else {
            input = torch::randint(-10, 10, shape_input, options);
            tensor1 = torch::randint(-10, 10, shape_tensor1, options);
            tensor2 = torch::randint(-10, 10, shape_tensor2, options);
        }
        
        // Test with various edge cases
        uint8_t edge_case = 0;
        if (consumeBytes(data, offset, size, edge_case)) {
            switch (edge_case % 8) {
                case 0:  // Normal case
                    break;
                case 1:  // Zero tensor
                    input.zero_();
                    break;
                case 2:  // NaN/Inf for float types
                    if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                        tensor1[0] = std::numeric_limits<float>::infinity();
                    }
                    break;
                case 3:  // Empty tensor (if possible)
                    if (rank > 0 && shape_input.size() > 0) {
                        shape_input[0] = 0;
                        input = torch::empty(shape_input, options);
                    }
                    break;
                case 4:  // Scalar tensors
                    input = torch::randn({}, options);
                    tensor1 = torch::randn({}, options);
                    tensor2 = torch::randn({}, options);
                    break;
                case 5:  // Mixed broadcasting shapes
                    if (rank > 1) {
                        shape_tensor1[rank-1] = 1;  // Make last dim 1 for broadcasting
                        tensor1 = torch::randn(shape_tensor1, options);
                    }
                    break;
                case 6:  // Contiguous vs non-contiguous
                    if (input.numel() > 1) {
                        input = input.transpose(0, -1);
                    }
                    break;
                case 7:  // Very small values
                    if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                        tensor1.mul_(1e-30);
                        tensor2.mul_(1e-30);
                    }
                    break;
            }
        }
        
        // Prepare value parameter
        torch::Scalar value;
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            // Use float value for float tensors
            if (value_type % 3 == 0) {
                value = torch::Scalar(value_float);
            } else if (value_type % 3 == 1) {
                value = torch::Scalar(0.0f);
            } else {
                value = torch::Scalar(-value_float);
            }
        } else {
            // Use integer value for integer tensors
            value = torch::Scalar(value_int % 100 - 50);  // Range -50 to 49
        }
        
        // Execute addcmul operation
        torch::Tensor result;
        if (use_out % 2 == 0) {
            // Without out parameter
            result = torch::addcmul(input, tensor1, tensor2, value);
        } else {
            // With out parameter
            torch::Tensor out = torch::empty_like(input);
            torch::addcmul_out(out, input, tensor1, tensor2, value);
            result = out;
        }
        
        // Additional operations to increase coverage
        uint8_t post_op = 0;
        if (consumeBytes(data, offset, size, post_op)) {
            switch (post_op % 4) {
                case 0:
                    // In-place operation
                    input.addcmul_(tensor1, tensor2, value);
                    break;
                case 1:
                    // Check result properties
                    if (result.numel() > 0) {
                        auto sum = result.sum();
                        auto mean = result.mean();
                    }
                    break;
                case 2:
                    // Test with requires_grad
                    if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                        auto input_grad = input.requires_grad_(true);
                        auto result_grad = torch::addcmul(input_grad, tensor1, tensor2, value);
                        if (result_grad.requires_grad()) {
                            result_grad.sum().backward();
                        }
                    }
                    break;
                case 3:
                    // Test with different device (CPU only for fuzzing)
                    result = result.to(torch::kCPU);
                    break;
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid operations
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}