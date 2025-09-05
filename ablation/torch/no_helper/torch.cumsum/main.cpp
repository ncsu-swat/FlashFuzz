#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from the fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 16) return 0;  // Need minimum bytes for basic parameters
    
    size_t offset = 0;
    
    try {
        // Consume parameters for tensor creation
        uint8_t rank;
        if (!consumeBytes(data, size, offset, rank)) return 0;
        rank = (rank % 5) + 1;  // Limit rank to 1-5 dimensions
        
        // Build shape
        std::vector<int64_t> shape;
        for (uint8_t i = 0; i < rank; ++i) {
            uint8_t dim_size;
            if (!consumeBytes(data, size, offset, dim_size)) return 0;
            shape.push_back(static_cast<int64_t>(dim_size % 10));  // Limit dimension sizes
        }
        
        // Consume dtype selector
        uint8_t dtype_selector;
        if (!consumeBytes(data, size, offset, dtype_selector)) return 0;
        
        // Map to actual dtype
        torch::ScalarType dtype;
        switch (dtype_selector % 8) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kInt16; break;
            case 5: dtype = torch::kInt8; break;
            case 6: dtype = torch::kUInt8; break;
            case 7: dtype = torch::kFloat16; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Consume dimension for cumsum
        int8_t dim;
        if (!consumeBytes(data, size, offset, dim)) return 0;
        
        // Consume optional output dtype selector
        uint8_t out_dtype_selector;
        bool use_out_dtype = false;
        torch::ScalarType out_dtype = dtype;
        if (consumeBytes(data, size, offset, out_dtype_selector)) {
            if (out_dtype_selector % 2 == 0) {
                use_out_dtype = true;
                switch (out_dtype_selector % 8) {
                    case 0: out_dtype = torch::kFloat32; break;
                    case 1: out_dtype = torch::kFloat64; break;
                    case 2: out_dtype = torch::kInt32; break;
                    case 3: out_dtype = torch::kInt64; break;
                    case 4: out_dtype = torch::kInt16; break;
                    case 5: out_dtype = torch::kInt8; break;
                    case 6: out_dtype = torch::kUInt8; break;
                    case 7: out_dtype = torch::kFloat16; break;
                    default: out_dtype = torch::kFloat32; break;
                }
            }
        }
        
        // Create input tensor
        torch::Tensor input;
        int64_t numel = 1;
        for (auto s : shape) {
            numel *= s;
        }
        
        if (numel == 0) {
            // Handle empty tensor case
            input = torch::empty(shape, torch::dtype(dtype));
        } else if (numel > 100000) {
            // Limit total elements to prevent OOM
            return 0;
        } else {
            // Fill tensor with fuzzed data
            input = torch::empty(shape, torch::dtype(dtype));
            
            // Try to fill with remaining bytes
            size_t remaining = size - offset;
            if (remaining > 0) {
                if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                    input = torch::randn(shape, torch::dtype(dtype));
                    // Add some variation based on fuzzer data
                    if (remaining >= sizeof(float)) {
                        float scale;
                        consumeBytes(data, size, offset, scale);
                        input = input * (1.0f + std::fmod(scale, 10.0f));
                    }
                } else {
                    input = torch::randint(-100, 100, shape, torch::dtype(dtype));
                }
            }
        }
        
        // Test cumsum with different configurations
        torch::Tensor result;
        
        // Test 1: Basic cumsum
        if (rank > 0) {
            int64_t actual_dim = dim % rank;
            if (actual_dim < 0) actual_dim += rank;
            
            result = torch::cumsum(input, actual_dim);
            
            // Test 2: With dtype conversion
            if (use_out_dtype) {
                result = torch::cumsum(input, actual_dim, out_dtype);
            }
            
            // Test 3: With pre-allocated output tensor
            uint8_t use_out;
            if (consumeBytes(data, size, offset, use_out) && (use_out % 3 == 0)) {
                torch::Tensor out = torch::empty_like(input);
                torch::cumsum_out(out, input, actual_dim);
            }
        }
        
        // Test edge cases
        if (shape.size() > 0 && shape[0] == 0) {
            // Empty tensor along first dimension
            result = torch::cumsum(input, 0);
        }
        
        // Test negative dimensions
        if (rank > 0) {
            int64_t neg_dim = -(dim % rank) - 1;
            if (neg_dim >= -static_cast<int64_t>(rank) && neg_dim < 0) {
                result = torch::cumsum(input, neg_dim);
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