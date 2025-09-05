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
    if (size < 4) return 0;
    
    size_t offset = 0;
    
    try {
        // Consume configuration bytes
        uint8_t config1, config2, config3, config4;
        if (!consumeBytes(data, offset, size, config1)) return 0;
        if (!consumeBytes(data, offset, size, config2)) return 0;
        if (!consumeBytes(data, offset, size, config3)) return 0;
        if (!consumeBytes(data, offset, size, config4)) return 0;
        
        // Determine tensor properties from config bytes
        bool use_scalar_other = (config1 & 1);
        bool use_out_tensor = (config1 & 2);
        bool broadcast_test = (config1 & 4);
        
        // Select dtype
        int dtype_idx = (config2 % 6);
        torch::ScalarType dtype;
        switch(dtype_idx) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kInt8; break;
            case 5: dtype = torch::kComplexFloat; break;
            default: dtype = torch::kFloat32;
        }
        
        // Determine tensor shapes
        std::vector<int64_t> shape1, shape2;
        
        if (broadcast_test) {
            // Test broadcasting scenarios
            uint8_t broadcast_case = config3 % 5;
            switch(broadcast_case) {
                case 0: // scalar x tensor
                    shape1 = {1};
                    shape2 = {3, 4};
                    break;
                case 1: // different but compatible shapes
                    shape1 = {4, 1};
                    shape2 = {1, 4};
                    break;
                case 2: // batch dimension broadcasting
                    shape1 = {1, 3, 3};
                    shape2 = {5, 1, 3};
                    break;
                case 3: // edge case: empty tensor
                    shape1 = {0};
                    shape2 = {0};
                    break;
                case 4: // single element tensors
                    shape1 = {1, 1, 1};
                    shape2 = {1};
                    break;
            }
        } else {
            // Generate random shapes from fuzzer data
            uint8_t ndim1 = (config3 % 4) + 1;
            uint8_t ndim2 = (config4 % 4) + 1;
            
            for (int i = 0; i < ndim1; i++) {
                uint8_t dim_size;
                if (!consumeBytes(data, offset, size, dim_size)) {
                    dim_size = (i + 1) * 2;
                }
                shape1.push_back((dim_size % 10) + 1);
            }
            
            if (!use_scalar_other) {
                for (int i = 0; i < ndim2; i++) {
                    uint8_t dim_size;
                    if (!consumeBytes(data, offset, size, dim_size)) {
                        dim_size = (i + 1) * 3;
                    }
                    shape2.push_back((dim_size % 10) + 1);
                }
            }
        }
        
        // Create input tensor
        torch::Tensor input = torch::randn(shape1, torch::dtype(dtype));
        
        // Fill with fuzzer data if available
        if (offset < size && dtype != torch::kComplexFloat) {
            size_t tensor_bytes = input.numel() * input.element_size();
            size_t available = size - offset;
            if (available > 0) {
                size_t to_copy = std::min(tensor_bytes, available);
                std::memcpy(input.data_ptr(), data + offset, to_copy);
                offset += to_copy;
            }
        }
        
        torch::Tensor result;
        
        if (use_scalar_other) {
            // Test multiplication with scalar
            float scalar_value;
            if (!consumeBytes(data, offset, size, scalar_value)) {
                scalar_value = 2.5f;
            }
            
            if (use_out_tensor) {
                torch::Tensor out = torch::empty_like(input);
                result = torch::mul(input, scalar_value, out);
            } else {
                result = torch::mul(input, scalar_value);
            }
        } else {
            // Test multiplication with tensor
            torch::Tensor other;
            
            if (shape2.empty()) {
                other = torch::randn({1}, torch::dtype(dtype));
            } else {
                other = torch::randn(shape2, torch::dtype(dtype));
            }
            
            // Fill other tensor with fuzzer data if available
            if (offset < size && dtype != torch::kComplexFloat) {
                size_t tensor_bytes = other.numel() * other.element_size();
                size_t available = size - offset;
                if (available > 0) {
                    size_t to_copy = std::min(tensor_bytes, available);
                    std::memcpy(other.data_ptr(), data + offset, to_copy);
                }
            }
            
            if (use_out_tensor) {
                // Determine output shape for broadcasting
                try {
                    auto broadcast_shape = torch::broadcast_shapes({input.sizes().vec(), other.sizes().vec()});
                    torch::Tensor out = torch::empty(broadcast_shape, torch::dtype(dtype));
                    result = torch::mul(input, other, out);
                } catch (...) {
                    // If broadcasting fails, try without out tensor
                    result = torch::mul(input, other);
                }
            } else {
                result = torch::mul(input, other);
            }
        }
        
        // Additional operations to exercise more code paths
        if (result.numel() > 0) {
            // Test in-place multiplication
            torch::Tensor temp = input.clone();
            if (use_scalar_other) {
                temp.mul_(2.0);
            } else if (input.sizes() == result.sizes()) {
                temp.mul_(result);
            }
            
            // Test chained operations
            auto chained = torch::mul(torch::mul(input, 2.0), 0.5);
            
            // Test with different memory layouts
            if (input.dim() > 1) {
                auto transposed = input.transpose(0, -1);
                auto trans_result = torch::mul(transposed, 3.14);
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