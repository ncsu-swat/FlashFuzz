#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
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
    if (size < 16) return 0;  // Need minimum bytes for basic configuration
    
    try {
        size_t offset = 0;
        
        // Consume configuration bytes
        uint8_t tensor_config1, tensor_config2;
        uint8_t dtype_selector1, dtype_selector2;
        uint8_t dim_count1, dim_count2;
        uint8_t scalar_mode;  // 0: both tensors, 1: first scalar, 2: second scalar
        uint8_t device_type;
        
        if (!consumeBytes(data, offset, size, tensor_config1)) return 0;
        if (!consumeBytes(data, offset, size, tensor_config2)) return 0;
        if (!consumeBytes(data, offset, size, dtype_selector1)) return 0;
        if (!consumeBytes(data, offset, size, dtype_selector2)) return 0;
        if (!consumeBytes(data, offset, size, dim_count1)) return 0;
        if (!consumeBytes(data, offset, size, dim_count2)) return 0;
        if (!consumeBytes(data, offset, size, scalar_mode)) return 0;
        if (!consumeBytes(data, offset, size, device_type)) return 0;
        
        // Map dtype selectors to actual dtypes (focus on floating point for zeta)
        auto getDtype = [](uint8_t selector) {
            switch (selector % 4) {
                case 0: return torch::kFloat32;
                case 1: return torch::kFloat64;
                case 2: return torch::kFloat16;
                case 3: return torch::kBFloat16;
                default: return torch::kFloat32;
            }
        };
        
        auto dtype1 = getDtype(dtype_selector1);
        auto dtype2 = getDtype(dtype_selector2);
        
        // Determine device
        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available() && (device_type % 2 == 1)) {
            device = torch::kCUDA;
        }
        
        // Build tensor shapes
        dim_count1 = (dim_count1 % 5) + 1;  // 1-5 dimensions
        dim_count2 = (dim_count2 % 5) + 1;
        
        std::vector<int64_t> shape1, shape2;
        for (uint8_t i = 0; i < dim_count1; ++i) {
            uint8_t dim_size;
            if (!consumeBytes(data, offset, size, dim_size)) {
                dim_size = 1;
            }
            shape1.push_back((dim_size % 10) + 1);  // 1-10 per dimension
        }
        
        for (uint8_t i = 0; i < dim_count2; ++i) {
            uint8_t dim_size;
            if (!consumeBytes(data, offset, size, dim_size)) {
                dim_size = 1;
            }
            shape2.push_back((dim_size % 10) + 1);
        }
        
        // Create tensors based on scalar_mode
        torch::Tensor input_tensor, other_tensor;
        scalar_mode = scalar_mode % 4;
        
        if (scalar_mode == 0 || scalar_mode == 2) {
            // First argument is tensor
            input_tensor = torch::randn(shape1, torch::TensorOptions().dtype(dtype1).device(device));
            
            // Fill with fuzzer data if available
            if (offset < size) {
                float scale;
                if (consumeBytes(data, offset, size, scale)) {
                    scale = std::fmod(scale, 100.0f) - 50.0f;  // Range [-50, 50]
                    input_tensor = input_tensor * scale;
                }
                uint8_t add_value;
                if (consumeBytes(data, offset, size, add_value)) {
                    input_tensor = input_tensor + (add_value % 10);
                }
            }
        }
        
        if (scalar_mode == 0 || scalar_mode == 1) {
            // Second argument is tensor
            other_tensor = torch::randn(shape2, torch::TensorOptions().dtype(dtype2).device(device));
            
            // Fill with fuzzer data if available
            if (offset < size) {
                float scale;
                if (consumeBytes(data, offset, size, scale)) {
                    scale = std::fmod(scale, 20.0f) + 0.1f;  // Positive values for q
                    other_tensor = torch::abs(other_tensor) * scale;
                }
                uint8_t add_value;
                if (consumeBytes(data, offset, size, add_value)) {
                    other_tensor = other_tensor + (add_value % 5) + 1;  // Ensure positive
                }
            }
        }
        
        // Call torch.special.zeta with different configurations
        torch::Tensor result;
        
        if (scalar_mode == 0) {
            // Both tensors
            result = torch::special::zeta(input_tensor, other_tensor);
        } else if (scalar_mode == 1) {
            // First is scalar
            float scalar_val = 2.0f;
            if (offset < size) {
                consumeBytes(data, offset, size, scalar_val);
                scalar_val = std::fmod(std::abs(scalar_val), 10.0f) + 1.0f;
            }
            result = torch::special::zeta(scalar_val, other_tensor);
        } else if (scalar_mode == 2) {
            // Second is scalar
            float scalar_val = 1.0f;
            if (offset < size) {
                consumeBytes(data, offset, size, scalar_val);
                scalar_val = std::fmod(std::abs(scalar_val), 10.0f) + 0.1f;
            }
            result = torch::special::zeta(input_tensor, scalar_val);
        } else {
            // Both scalars
            float scalar1 = 2.0f, scalar2 = 1.0f;
            if (offset + 8 <= size) {
                consumeBytes(data, offset, size, scalar1);
                consumeBytes(data, offset, size, scalar2);
                scalar1 = std::fmod(std::abs(scalar1), 10.0f) + 1.0f;
                scalar2 = std::fmod(std::abs(scalar2), 10.0f) + 0.1f;
            }
            result = torch::special::zeta(scalar1, scalar2);
        }
        
        // Test with output tensor
        uint8_t use_out;
        if (consumeBytes(data, offset, size, use_out) && (use_out % 3 == 0)) {
            torch::Tensor out_tensor = torch::empty_like(result);
            torch::special::zeta_out(out_tensor, 
                                    scalar_mode >= 2 ? torch::tensor(2.0f) : input_tensor,
                                    scalar_mode == 1 || scalar_mode == 3 ? torch::tensor(1.0f) : other_tensor);
        }
        
        // Force computation
        if (result.numel() > 0) {
            result.sum().item<float>();
        }
        
    } catch (const c10::Error& e) {
        // PyTorch errors are expected for invalid inputs
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