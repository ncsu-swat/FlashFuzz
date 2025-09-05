#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t*& data, size_t& size, T& value) {
    if (size < sizeof(T)) return false;
    std::memcpy(&value, data, sizeof(T));
    data += sizeof(T);
    size -= sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0;
    
    try {
        // Consume configuration bytes
        uint8_t dtype_choice = 0;
        uint8_t use_N = 0;
        uint8_t num_batch_dims = 0;
        uint8_t vector_size = 0;
        
        if (!consumeBytes(data, size, dtype_choice)) return 0;
        if (!consumeBytes(data, size, use_N)) return 0;
        if (!consumeBytes(data, size, num_batch_dims)) return 0;
        if (!consumeBytes(data, size, vector_size)) return 0;
        
        // Limit dimensions to prevent OOM
        dtype_choice = dtype_choice % 7;  // 7 dtype options
        num_batch_dims = num_batch_dims % 4;  // Max 3 batch dimensions
        vector_size = (vector_size % 100) + 1;  // Vector size 1-100
        
        // Select dtype
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kComplexFloat; break;
            case 3: dtype = torch::kComplexDouble; break;
            case 4: dtype = torch::kInt32; break;
            case 5: dtype = torch::kInt64; break;
            case 6: dtype = torch::kInt8; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Build tensor shape
        std::vector<int64_t> shape;
        for (uint8_t i = 0; i < num_batch_dims; ++i) {
            uint8_t dim_size = 0;
            if (!consumeBytes(data, size, dim_size)) {
                dim_size = 2;  // Default batch dimension size
            }
            shape.push_back((dim_size % 10) + 1);  // Batch dims 1-10
        }
        shape.push_back(vector_size);
        
        // Calculate total elements needed
        int64_t total_elements = 1;
        for (auto dim : shape) {
            total_elements *= dim;
        }
        
        // Create input tensor with fuzzer data
        torch::Tensor x;
        
        if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) {
            // For complex types, create real and imaginary parts
            int64_t real_elements = total_elements;
            std::vector<float> real_data, imag_data;
            
            for (int64_t i = 0; i < real_elements; ++i) {
                float real_val = 0.0f, imag_val = 0.0f;
                if (size >= sizeof(float)) {
                    consumeBytes(data, size, real_val);
                } else {
                    real_val = static_cast<float>(i) / 10.0f;
                }
                if (size >= sizeof(float)) {
                    consumeBytes(data, size, imag_val);
                } else {
                    imag_val = static_cast<float>(i) / 20.0f;
                }
                real_data.push_back(real_val);
                imag_data.push_back(imag_val);
            }
            
            auto real_tensor = torch::from_blob(real_data.data(), shape, torch::kFloat32).clone();
            auto imag_tensor = torch::from_blob(imag_data.data(), shape, torch::kFloat32).clone();
            x = torch::complex(real_tensor, imag_tensor);
            if (dtype == torch::kComplexDouble) {
                x = x.to(torch::kComplexDouble);
            }
        } else if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            // For floating point types
            std::vector<float> float_data;
            for (int64_t i = 0; i < total_elements; ++i) {
                float val = 0.0f;
                if (size >= sizeof(float)) {
                    consumeBytes(data, size, val);
                } else {
                    val = static_cast<float>(i) / 10.0f;
                }
                float_data.push_back(val);
            }
            x = torch::from_blob(float_data.data(), shape, torch::kFloat32).clone().to(dtype);
        } else {
            // For integral types
            std::vector<int32_t> int_data;
            for (int64_t i = 0; i < total_elements; ++i) {
                int32_t val = 0;
                if (size >= sizeof(int32_t)) {
                    consumeBytes(data, size, val);
                } else {
                    val = i;
                }
                int_data.push_back(val);
            }
            x = torch::from_blob(int_data.data(), shape, torch::kInt32).clone().to(dtype);
        }
        
        // Call torch.linalg.vander with optional N parameter
        torch::Tensor result;
        if ((use_N % 2) == 1 && size >= sizeof(uint8_t)) {
            uint8_t N_value = 0;
            consumeBytes(data, size, N_value);
            int64_t N = (N_value % 150) + 1;  // N from 1 to 150
            
            // Call with explicit N
            result = torch::linalg::vander(x, N);
        } else {
            // Call without N (uses x.size(-1) as default)
            result = torch::linalg::vander(x);
        }
        
        // Additional operations to increase coverage
        if (result.numel() > 0) {
            // Test flip operation as mentioned in docs
            auto flipped = result.flip({-1});
            
            // Test with edge cases
            if (vector_size == 1) {
                // Single element vector case
                auto sum_val = result.sum();
            }
            
            // Test batch processing
            if (num_batch_dims > 0) {
                auto batch_sum = result.sum({-1});
            }
        }
        
        // Test with zero-dimensional edge case
        if (size > 0 && (data[0] % 20) == 0) {
            auto empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(dtype));
            try {
                auto empty_result = torch::linalg::vander(empty_tensor);
            } catch (...) {
                // Expected to potentially fail with empty input
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid inputs
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Unknown exceptions
        return -1;
    }
    
    return 0;
}