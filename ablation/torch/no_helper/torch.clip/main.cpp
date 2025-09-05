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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        if (size < 16) return 0;  // Need minimum bytes for basic parameters
        
        size_t offset = 0;
        
        // Consume configuration bytes
        uint8_t dtype_selector = 0;
        uint8_t ndims = 0;
        uint8_t has_min = 0;
        uint8_t has_max = 0;
        uint8_t use_scalar_min = 0;
        uint8_t use_scalar_max = 0;
        
        consumeBytes(data, size, offset, dtype_selector);
        consumeBytes(data, size, offset, ndims);
        consumeBytes(data, size, offset, has_min);
        consumeBytes(data, size, offset, has_max);
        consumeBytes(data, size, offset, use_scalar_min);
        consumeBytes(data, size, offset, use_scalar_max);
        
        // Limit dimensions to reasonable range
        ndims = (ndims % 5) + 1;  // 1-5 dimensions
        
        // Build shape vector
        std::vector<int64_t> shape;
        for (size_t i = 0; i < ndims; ++i) {
            uint8_t dim_size = 0;
            if (!consumeBytes(data, size, offset, dim_size)) {
                dim_size = 1;
            }
            // Allow 0-sized dimensions for edge cases
            shape.push_back(dim_size % 10);  // 0-9 per dimension
        }
        
        // Select dtype based on selector
        torch::ScalarType dtype;
        switch (dtype_selector % 6) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kInt8; break;
            case 5: dtype = torch::kBool; break;
            default: dtype = torch::kFloat32;
        }
        
        // Create input tensor
        torch::Tensor input;
        try {
            auto options = torch::TensorOptions().dtype(dtype);
            
            // Calculate total elements
            int64_t total_elements = 1;
            for (auto dim : shape) {
                total_elements *= dim;
            }
            
            if (total_elements == 0) {
                // Create empty tensor
                input = torch::empty(shape, options);
            } else if (total_elements > 1000000) {
                // Limit size to prevent OOM
                input = torch::randn({10, 10}, options);
            } else {
                // Fill tensor with fuzzer data or random values
                if (offset + total_elements * 4 <= size) {
                    // Use fuzzer data
                    std::vector<float> values;
                    for (int64_t i = 0; i < total_elements; ++i) {
                        float val = 0;
                        consumeBytes(data, size, offset, val);
                        values.push_back(val);
                    }
                    auto temp = torch::from_blob(values.data(), shape, torch::kFloat32).clone();
                    input = temp.to(dtype);
                } else {
                    // Use random values
                    input = torch::randn(shape, options);
                }
            }
        } catch (...) {
            // Fallback to simple tensor
            input = torch::randn({2, 3}, torch::kFloat32);
        }
        
        // Prepare min/max values
        torch::Tensor result;
        
        if ((has_min % 2) && (has_max % 2)) {
            // Both min and max
            if ((use_scalar_min % 2) && (use_scalar_max % 2)) {
                // Both scalars
                float min_val = 0, max_val = 0;
                consumeBytes(data, size, offset, min_val);
                consumeBytes(data, size, offset, max_val);
                result = torch::clip(input, min_val, max_val);
            } else if (use_scalar_min % 2) {
                // Min scalar, max tensor
                float min_val = 0;
                consumeBytes(data, size, offset, min_val);
                torch::Tensor max_tensor = torch::randn_like(input);
                result = torch::clip(input, min_val, max_tensor);
            } else if (use_scalar_max % 2) {
                // Min tensor, max scalar
                torch::Tensor min_tensor = torch::randn_like(input);
                float max_val = 0;
                consumeBytes(data, size, offset, max_val);
                result = torch::clip(input, min_tensor, max_val);
            } else {
                // Both tensors
                torch::Tensor min_tensor = torch::randn_like(input);
                torch::Tensor max_tensor = torch::randn_like(input);
                result = torch::clip(input, min_tensor, max_tensor);
            }
        } else if (has_min % 2) {
            // Only min
            if (use_scalar_min % 2) {
                float min_val = 0;
                consumeBytes(data, size, offset, min_val);
                result = torch::clip(input, min_val, c10::nullopt);
            } else {
                torch::Tensor min_tensor = torch::randn_like(input);
                result = torch::clip(input, min_tensor, c10::nullopt);
            }
        } else if (has_max % 2) {
            // Only max
            if (use_scalar_max % 2) {
                float max_val = 0;
                consumeBytes(data, size, offset, max_val);
                result = torch::clip(input, c10::nullopt, max_val);
            } else {
                torch::Tensor max_tensor = torch::randn_like(input);
                result = torch::clip(input, c10::nullopt, max_tensor);
            }
        } else {
            // Neither min nor max (edge case)
            result = torch::clip(input, c10::nullopt, c10::nullopt);
        }
        
        // Try in-place operation
        if (offset < size && data[offset] % 2) {
            torch::Tensor input_copy = input.clone();
            input_copy.clip_(c10::nullopt, c10::nullopt);
        }
        
        // Force computation
        if (result.numel() > 0) {
            result.sum().item<float>();
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cout << "Exception caught: unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}