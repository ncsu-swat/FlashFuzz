#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 16) {
        return 0;  // Need minimum bytes for basic parameters
    }

    try {
        size_t offset = 0;
        
        // Extract basic parameters from fuzzer input
        uint8_t rank = data[offset++] % 5 + 1;  // tensor rank 1-5
        uint8_t dtype_selector = data[offset++] % 6;  // select dtype
        bool keepdim = data[offset++] & 1;
        uint8_t dim_count = data[offset++] % rank + 1;  // number of dims to reduce
        bool use_out_tensor = data[offset++] & 1;
        
        // Build tensor shape
        std::vector<int64_t> shape;
        for (size_t i = 0; i < rank && offset < size; ++i) {
            int64_t dim_size = (data[offset++] % 8) + 1;  // dimensions 1-8
            shape.push_back(dim_size);
        }
        
        // Handle edge case: empty shape
        if (offset >= size && shape.empty()) {
            shape.push_back(1);
        }
        
        // Build reduction dimensions
        std::vector<int64_t> dims;
        std::vector<bool> dim_used(rank, false);
        for (size_t i = 0; i < dim_count && offset < size; ++i) {
            int64_t dim = data[offset++] % rank;
            if (!dim_used[dim]) {
                dims.push_back(dim);
                dim_used[dim] = true;
            }
        }
        
        // Ensure at least one dimension if empty
        if (dims.empty()) {
            dims.push_back(0);
        }
        
        // Select dtype
        torch::ScalarType dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kFloat16; break;
            case 5: dtype = torch::kBFloat16; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Create input tensor with remaining bytes as data
        torch::Tensor input;
        size_t remaining = size - offset;
        
        if (remaining > 0) {
            // Use fuzzer data to initialize tensor values
            auto options = torch::TensorOptions().dtype(dtype);
            input = torch::empty(shape, options);
            
            // Fill tensor with fuzzer data
            size_t element_size = input.element_size();
            size_t total_elements = input.numel();
            size_t bytes_needed = total_elements * element_size;
            
            if (bytes_needed > 0) {
                void* tensor_data = input.data_ptr();
                size_t bytes_to_copy = std::min(remaining, bytes_needed);
                std::memcpy(tensor_data, data + offset, bytes_to_copy);
                
                // Fill remaining with pattern if not enough data
                if (bytes_to_copy < bytes_needed) {
                    uint8_t* ptr = static_cast<uint8_t*>(tensor_data) + bytes_to_copy;
                    for (size_t i = bytes_to_copy; i < bytes_needed; ++i) {
                        *ptr++ = data[offset + (i % remaining)];
                    }
                }
            }
        } else {
            // Create random tensor if no data left
            input = torch::randn(shape, torch::TensorOptions().dtype(dtype));
        }
        
        // Test various edge cases
        if ((data[0] & 0x3) == 0 && input.numel() > 0) {
            // Sometimes add NaN/Inf values for floating point tensors
            if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                auto flat = input.flatten();
                if (flat.numel() > 0) {
                    flat[0] = std::numeric_limits<float>::quiet_NaN();
                    if (flat.numel() > 1) {
                        flat[1] = std::numeric_limits<float>::infinity();
                    }
                    if (flat.numel() > 2) {
                        flat[2] = -std::numeric_limits<float>::infinity();
                    }
                }
            }
        }
        
        // Perform amax operation
        torch::Tensor result;
        
        if (use_out_tensor && !dims.empty()) {
            // Calculate output shape for out tensor
            std::vector<int64_t> out_shape = shape;
            if (keepdim) {
                for (auto d : dims) {
                    if (d >= 0 && d < static_cast<int64_t>(out_shape.size())) {
                        out_shape[d] = 1;
                    }
                }
            } else {
                // Remove reduced dimensions (in reverse order to maintain indices)
                std::sort(dims.begin(), dims.end(), std::greater<int64_t>());
                for (auto d : dims) {
                    if (d >= 0 && d < static_cast<int64_t>(out_shape.size())) {
                        out_shape.erase(out_shape.begin() + d);
                    }
                }
                if (out_shape.empty()) {
                    out_shape.push_back(1);  // scalar case
                }
            }
            
            torch::Tensor out = torch::empty(out_shape, input.options());
            result = torch::amax_out(out, input, dims, keepdim);
        } else {
            result = torch::amax(input, dims, keepdim);
        }
        
        // Additional operations to exercise more paths
        if (result.defined() && result.numel() > 0) {
            // Test backward pass if applicable
            if (input.requires_grad() && input.is_floating_point()) {
                input.set_requires_grad(true);
                auto res2 = torch::amax(input, dims, keepdim);
                if (res2.numel() == 1) {
                    res2.backward();
                } else {
                    auto grad = torch::ones_like(res2);
                    res2.backward(grad);
                }
            }
            
            // Test with negative dimensions
            if (!dims.empty() && (data[1] & 1)) {
                std::vector<int64_t> neg_dims;
                for (auto d : dims) {
                    neg_dims.push_back(d - rank);
                }
                auto res3 = torch::amax(input, neg_dims, keepdim);
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