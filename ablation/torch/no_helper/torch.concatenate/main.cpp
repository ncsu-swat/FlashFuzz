#include <torch/torch.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <iostream>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        if (size < 4) return 0;
        
        size_t offset = 0;
        
        // Consume number of tensors to concatenate (1-10)
        uint8_t num_tensors_raw;
        if (!consumeBytes(data, size, offset, num_tensors_raw)) return 0;
        int num_tensors = 1 + (num_tensors_raw % 10);
        
        // Consume axis
        int8_t axis;
        if (!consumeBytes(data, size, offset, axis)) return 0;
        
        // Consume dtype selector
        uint8_t dtype_selector;
        if (!consumeBytes(data, size, offset, dtype_selector)) return 0;
        
        std::vector<torch::Tensor> tensors;
        tensors.reserve(num_tensors);
        
        // Create tensors with varying properties
        for (int i = 0; i < num_tensors; i++) {
            if (offset >= size) break;
            
            // Consume number of dimensions (0-5)
            uint8_t ndim_raw;
            if (!consumeBytes(data, size, offset, ndim_raw)) break;
            int ndim = ndim_raw % 6;
            
            std::vector<int64_t> shape;
            bool has_zero_dim = false;
            
            for (int d = 0; d < ndim; d++) {
                uint8_t dim_size_raw;
                if (!consumeBytes(data, size, offset, dim_size_raw)) {
                    shape.push_back(1);
                    continue;
                }
                // Allow dimensions 0-10
                int64_t dim_size = dim_size_raw % 11;
                shape.push_back(dim_size);
                if (dim_size == 0) has_zero_dim = true;
            }
            
            // Determine dtype based on selector
            torch::ScalarType dtype;
            switch (dtype_selector % 8) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kInt32; break;
                case 3: dtype = torch::kInt64; break;
                case 4: dtype = torch::kInt8; break;
                case 5: dtype = torch::kUInt8; break;
                case 6: dtype = torch::kBool; break;
                case 7: dtype = torch::kFloat16; break;
                default: dtype = torch::kFloat32; break;
            }
            
            // Create tensor with various initialization strategies
            torch::Tensor tensor;
            uint8_t init_strategy;
            if (!consumeBytes(data, size, offset, init_strategy)) {
                init_strategy = 0;
            }
            
            try {
                switch (init_strategy % 6) {
                    case 0:
                        tensor = torch::zeros(shape, torch::dtype(dtype));
                        break;
                    case 1:
                        tensor = torch::ones(shape, torch::dtype(dtype));
                        break;
                    case 2:
                        tensor = torch::randn(shape, torch::dtype(dtype));
                        break;
                    case 3:
                        tensor = torch::empty(shape, torch::dtype(dtype));
                        break;
                    case 4:
                        if (!has_zero_dim && shape.size() > 0) {
                            tensor = torch::arange(shape[0], torch::dtype(dtype));
                            if (shape.size() > 1) {
                                std::vector<int64_t> new_shape(shape.begin() + 1, shape.end());
                                new_shape.insert(new_shape.begin(), shape[0]);
                                tensor = tensor.reshape(new_shape);
                            }
                        } else {
                            tensor = torch::zeros(shape, torch::dtype(dtype));
                        }
                        break;
                    case 5:
                        tensor = torch::full(shape, 42, torch::dtype(dtype));
                        break;
                    default:
                        tensor = torch::zeros(shape, torch::dtype(dtype));
                        break;
                }
                
                // Optionally make tensor non-contiguous
                uint8_t make_noncontig;
                if (consumeBytes(data, size, offset, make_noncontig) && (make_noncontig % 4 == 0)) {
                    if (tensor.dim() >= 2 && tensor.size(0) > 0 && tensor.size(1) > 0) {
                        tensor = tensor.transpose(0, 1);
                    }
                }
                
                // Optionally add requires_grad
                uint8_t requires_grad;
                if (consumeBytes(data, size, offset, requires_grad) && (requires_grad % 3 == 0)) {
                    if (tensor.dtype() == torch::kFloat32 || tensor.dtype() == torch::kFloat64) {
                        tensor.requires_grad_(true);
                    }
                }
                
                tensors.push_back(tensor);
            } catch (const c10::Error& e) {
                // If tensor creation fails, add a simple fallback tensor
                tensors.push_back(torch::zeros({1}, torch::dtype(dtype)));
            }
        }
        
        if (tensors.empty()) {
            tensors.push_back(torch::zeros({1}));
        }
        
        // Try concatenation with the generated tensors and axis
        try {
            torch::Tensor result = torch::cat(tensors, axis);
            
            // Exercise the result tensor
            if (result.numel() > 0) {
                auto sum = result.sum();
                auto mean = result.mean();
                if (result.dim() > 0) {
                    auto shape = result.sizes();
                    auto strides = result.strides();
                }
            }
            
            // Try with explicit output tensor
            uint8_t use_out;
            if (consumeBytes(data, size, offset, use_out) && (use_out % 5 == 0)) {
                torch::Tensor out_tensor = torch::empty_like(result);
                torch::cat_out(out_tensor, tensors, axis);
            }
            
        } catch (const c10::Error& e) {
            // Concatenation might fail for valid reasons (incompatible shapes, invalid axis)
            // This is expected behavior we want to test
        }
        
        // Also test edge cases with empty tensor list
        if (offset < size) {
            uint8_t test_empty;
            if (consumeBytes(data, size, offset, test_empty) && (test_empty % 20 == 0)) {
                try {
                    std::vector<torch::Tensor> empty_list;
                    torch::cat(empty_list, 0);
                } catch (const c10::Error& e) {
                    // Expected to fail
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}