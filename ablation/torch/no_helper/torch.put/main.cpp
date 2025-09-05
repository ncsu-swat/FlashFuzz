#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from the fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t& offset, size_t size, T& out) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&out, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

// Helper to create a tensor from fuzzer input
torch::Tensor createTensorFromBytes(const uint8_t* data, size_t& offset, size_t size) {
    // Consume dtype selector
    uint8_t dtype_selector = 0;
    if (!consumeBytes(data, offset, size, dtype_selector)) {
        return torch::empty({0});
    }
    
    // Map to actual dtype
    torch::ScalarType dtype;
    switch (dtype_selector % 10) {
        case 0: dtype = torch::kFloat32; break;
        case 1: dtype = torch::kFloat64; break;
        case 2: dtype = torch::kInt32; break;
        case 3: dtype = torch::kInt64; break;
        case 4: dtype = torch::kInt16; break;
        case 5: dtype = torch::kInt8; break;
        case 6: dtype = torch::kUInt8; break;
        case 7: dtype = torch::kBool; break;
        case 8: dtype = torch::kFloat16; break;
        default: dtype = torch::kFloat32; break;
    }
    
    // Consume number of dimensions
    uint8_t num_dims = 0;
    if (!consumeBytes(data, offset, size, num_dims)) {
        return torch::empty({0}, torch::dtype(dtype));
    }
    num_dims = (num_dims % 5) + 1; // 1 to 5 dimensions
    
    // Consume shape
    std::vector<int64_t> shape;
    for (int i = 0; i < num_dims; i++) {
        uint8_t dim_size = 0;
        if (!consumeBytes(data, offset, size, dim_size)) {
            shape.push_back(1);
        } else {
            shape.push_back((dim_size % 10) + 1); // 1 to 10 per dimension
        }
    }
    
    // Create tensor with random data
    torch::Tensor tensor = torch::empty(shape, torch::dtype(dtype));
    
    // Fill with some fuzzer-controlled values
    int64_t numel = tensor.numel();
    if (numel > 0 && numel < 1000) { // Limit to avoid excessive memory
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            float val = 0.0f;
            if (consumeBytes(data, offset, size, val)) {
                tensor.fill_(val);
            }
        } else {
            int32_t val = 0;
            if (consumeBytes(data, offset, size, val)) {
                tensor.fill_(val);
            }
        }
    }
    
    return tensor;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 10) return 0; // Need minimum bytes
    
    try {
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input = createTensorFromBytes(data, offset, size);
        
        // Create index tensor - should be 1D long tensor
        uint8_t index_size = 0;
        if (!consumeBytes(data, offset, size, index_size)) {
            index_size = 1;
        }
        index_size = (index_size % 20) + 1; // 1 to 20 indices
        
        std::vector<int64_t> indices;
        for (int i = 0; i < index_size; i++) {
            int32_t idx = 0;
            if (consumeBytes(data, offset, size, idx)) {
                // Allow negative indices and out-of-bounds for edge cases
                indices.push_back(idx);
            } else {
                indices.push_back(i);
            }
        }
        torch::Tensor index = torch::tensor(indices, torch::kInt64);
        
        // Create source tensor (values to put)
        torch::Tensor source = createTensorFromBytes(data, offset, size);
        
        // Consume accumulate flag
        uint8_t accumulate_byte = 0;
        bool accumulate = false;
        if (consumeBytes(data, offset, size, accumulate_byte)) {
            accumulate = (accumulate_byte % 2) == 1;
        }
        
        // Try different variations of put operation
        uint8_t operation_type = 0;
        if (consumeBytes(data, offset, size, operation_type)) {
            operation_type = operation_type % 4;
        }
        
        torch::Tensor result;
        
        switch (operation_type) {
            case 0:
                // Basic put operation
                result = input.put(index, source, accumulate);
                break;
            case 1:
                // In-place put operation
                result = input.clone();
                result.put_(index, source, accumulate);
                break;
            case 2:
                // Put with flattened input
                if (input.numel() > 0) {
                    result = input.flatten().put(index, source, accumulate);
                }
                break;
            case 3:
                // Put with reshaped source
                if (source.numel() > 0 && index.numel() > 0) {
                    torch::Tensor reshaped_source = source.flatten();
                    if (reshaped_source.numel() >= index.numel()) {
                        reshaped_source = reshaped_source.narrow(0, 0, index.numel());
                    }
                    result = input.put(index, reshaped_source, accumulate);
                }
                break;
        }
        
        // Additional operations to increase coverage
        if (result.defined() && result.numel() > 0) {
            // Try accessing result
            auto shape = result.sizes();
            auto dtype = result.dtype();
            auto device = result.device();
            
            // Try some tensor operations on result
            if (result.numel() < 1000) {
                auto sum = result.sum();
                auto mean = result.to(torch::kFloat32).mean();
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid operations
        return 0;
    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}