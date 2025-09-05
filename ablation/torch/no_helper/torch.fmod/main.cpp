#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>

// Helper to consume bytes from the fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

// Create tensor from fuzzer input
torch::Tensor createTensorFromBytes(const uint8_t* data, size_t size, size_t& offset) {
    if (offset >= size) return torch::empty({0});
    
    // Consume dtype selector
    uint8_t dtype_selector = 0;
    if (!consumeBytes(data, size, offset, dtype_selector)) return torch::empty({0});
    
    // Consume rank
    uint8_t rank = 0;
    if (!consumeBytes(data, size, offset, rank)) return torch::empty({0});
    rank = (rank % 5) + 1; // Limit rank to 1-5
    
    // Consume shape
    std::vector<int64_t> shape;
    for (int i = 0; i < rank; i++) {
        uint8_t dim = 0;
        if (!consumeBytes(data, size, offset, dim)) break;
        shape.push_back((dim % 10) + 1); // Limit dimensions to 1-10
    }
    
    if (shape.empty()) shape.push_back(1);
    
    // Select dtype based on selector
    torch::ScalarType dtype;
    switch (dtype_selector % 6) {
        case 0: dtype = torch::kFloat32; break;
        case 1: dtype = torch::kFloat64; break;
        case 2: dtype = torch::kInt32; break;
        case 3: dtype = torch::kInt64; break;
        case 4: dtype = torch::kInt8; break;
        case 5: dtype = torch::kInt16; break;
        default: dtype = torch::kFloat32;
    }
    
    // Create tensor with random data
    torch::Tensor tensor = torch::empty(shape, torch::dtype(dtype));
    
    // Fill tensor with fuzzer data
    size_t num_elements = tensor.numel();
    if (num_elements > 0) {
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            // For floating point, create some special values
            uint8_t special_val = 0;
            if (consumeBytes(data, size, offset, special_val)) {
                switch (special_val % 5) {
                    case 0: tensor.fill_(0.0); break;
                    case 1: tensor.fill_(1.0); break;
                    case 2: tensor.fill_(-1.0); break;
                    case 3: tensor.fill_(std::numeric_limits<float>::infinity()); break;
                    case 4: tensor.fill_(std::numeric_limits<float>::quiet_NaN()); break;
                }
            }
            
            // Optionally add some random values
            for (int64_t i = 0; i < std::min((int64_t)10, num_elements); i++) {
                float val = 0;
                if (consumeBytes(data, size, offset, val)) {
                    tensor.view({-1})[i] = val;
                }
            }
        } else {
            // For integer types
            for (int64_t i = 0; i < std::min((int64_t)10, num_elements); i++) {
                int32_t val = 0;
                if (consumeBytes(data, size, offset, val)) {
                    tensor.view({-1})[i] = val;
                }
            }
        }
    }
    
    return tensor;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0;
    
    try {
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input = createTensorFromBytes(data, size, offset);
        
        // Decide whether to use tensor or scalar for 'other'
        uint8_t use_scalar = 0;
        consumeBytes(data, size, offset, use_scalar);
        
        torch::Tensor result;
        
        if (use_scalar % 2 == 0) {
            // Use scalar value
            float scalar_value = 0;
            consumeBytes(data, size, offset, scalar_value);
            
            // Handle special scalar values
            uint8_t special_scalar = 0;
            if (consumeBytes(data, size, offset, special_scalar)) {
                switch (special_scalar % 6) {
                    case 0: scalar_value = 0.0f; break;
                    case 1: scalar_value = 1.0f; break;
                    case 2: scalar_value = -1.0f; break;
                    case 3: scalar_value = 0.5f; break;
                    case 4: scalar_value = std::numeric_limits<float>::infinity(); break;
                    case 5: scalar_value = std::numeric_limits<float>::quiet_NaN(); break;
                }
            }
            
            // Test fmod with scalar
            result = torch::fmod(input, scalar_value);
        } else {
            // Use tensor for 'other'
            torch::Tensor other = createTensorFromBytes(data, size, offset);
            
            // Test fmod with tensor
            result = torch::fmod(input, other);
        }
        
        // Optionally test with out parameter
        uint8_t use_out = 0;
        if (consumeBytes(data, size, offset, use_out) && (use_out % 4 == 0)) {
            torch::Tensor out = torch::empty_like(result);
            torch::fmod_out(out, input, result);
        }
        
        // Force computation
        if (result.numel() > 0) {
            result.sum().item<float>();
        }
        
    } catch (const c10::Error& e) {
        // PyTorch errors are expected for invalid operations
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}