#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Helper to consume bytes from the fuzzer input
template<typename T>
bool consume(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        if (size < 4) return 0;
        
        size_t offset = 0;
        
        // Consume configuration bytes
        uint8_t tensor_dim;
        if (!consume(data, size, offset, tensor_dim)) return 0;
        tensor_dim = (tensor_dim % 3); // 0=scalar, 1=1D, 2=2D
        
        uint8_t dtype_choice;
        if (!consume(data, size, offset, dtype_choice)) return 0;
        dtype_choice = dtype_choice % 4;
        
        int8_t diagonal_offset;
        if (!consume(data, size, offset, diagonal_offset)) return 0;
        
        uint8_t use_out_tensor;
        if (!consume(data, size, offset, use_out_tensor)) return 0;
        
        // Determine dtype
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            default: dtype = torch::kFloat32;
        }
        
        torch::Tensor input;
        
        if (tensor_dim == 0) {
            // Scalar tensor (edge case)
            float scalar_val = 1.0f;
            consume(data, size, offset, scalar_val);
            input = torch::tensor(scalar_val, torch::dtype(dtype));
        } else if (tensor_dim == 1) {
            // 1D tensor (vector)
            uint8_t vec_size;
            if (!consume(data, size, offset, vec_size)) return 0;
            vec_size = (vec_size % 64) + 1; // Size between 1 and 64
            
            std::vector<float> vec_data;
            vec_data.reserve(vec_size);
            for (int i = 0; i < vec_size; i++) {
                float val = static_cast<float>(i);
                if (offset < size) {
                    uint8_t byte_val;
                    consume(data, size, offset, byte_val);
                    val = static_cast<float>(byte_val) / 128.0f - 1.0f;
                }
                vec_data.push_back(val);
            }
            input = torch::tensor(vec_data, torch::dtype(dtype));
        } else {
            // 2D tensor (matrix)
            uint8_t rows, cols;
            if (!consume(data, size, offset, rows)) return 0;
            if (!consume(data, size, offset, cols)) return 0;
            rows = (rows % 32) + 1; // Between 1 and 32
            cols = (cols % 32) + 1;
            
            std::vector<float> mat_data;
            mat_data.reserve(rows * cols);
            for (int i = 0; i < rows * cols; i++) {
                float val = static_cast<float>(i) / (rows * cols);
                if (offset < size) {
                    uint8_t byte_val;
                    consume(data, size, offset, byte_val);
                    val = static_cast<float>(byte_val) / 128.0f - 1.0f;
                }
                mat_data.push_back(val);
            }
            input = torch::tensor(mat_data, torch::dtype(dtype)).reshape({rows, cols});
        }
        
        // Test with different diagonal values
        int diagonal = static_cast<int>(diagonal_offset);
        
        // Test basic diag operation
        torch::Tensor result = torch::diag(input, diagonal);
        
        // Test with out parameter if requested
        if (use_out_tensor % 2 == 0) {
            // Determine expected output shape
            torch::Tensor out;
            if (input.dim() == 1) {
                // Vector input -> square matrix output
                int n = input.size(0);
                int out_size = n + std::abs(diagonal);
                out = torch::empty({out_size, out_size}, torch::dtype(dtype));
            } else if (input.dim() == 2) {
                // Matrix input -> vector output
                int rows = input.size(0);
                int cols = input.size(1);
                int diag_len;
                if (diagonal >= 0) {
                    diag_len = std::min(rows, cols - diagonal);
                } else {
                    diag_len = std::min(rows + diagonal, cols);
                }
                if (diag_len > 0) {
                    out = torch::empty({diag_len}, torch::dtype(dtype));
                } else {
                    out = torch::empty({0}, torch::dtype(dtype));
                }
            } else {
                // For other dimensions, let PyTorch handle it
                out = torch::empty({1}, torch::dtype(dtype));
            }
            
            // Call with out parameter
            torch::diag_out(out, input, diagonal);
        }
        
        // Test edge cases
        if (input.dim() == 2) {
            // Test with very large diagonal offsets
            torch::diag(input, 100);
            torch::diag(input, -100);
        }
        
        // Test with empty tensors
        if (offset % 7 == 0) {
            torch::Tensor empty_1d = torch::empty({0}, torch::dtype(dtype));
            torch::diag(empty_1d, diagonal);
            
            torch::Tensor empty_2d = torch::empty({0, 5}, torch::dtype(dtype));
            torch::diag(empty_2d, diagonal);
            
            torch::Tensor empty_2d_2 = torch::empty({5, 0}, torch::dtype(dtype));
            torch::diag(empty_2d_2, diagonal);
        }
        
        // Test with non-contiguous tensors
        if (input.dim() == 2 && input.size(0) > 1 && input.size(1) > 1) {
            torch::Tensor transposed = input.transpose(0, 1);
            torch::diag(transposed, diagonal);
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