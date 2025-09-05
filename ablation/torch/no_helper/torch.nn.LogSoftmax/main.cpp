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
    if (size < 4) return 0;
    
    size_t offset = 0;
    
    try {
        // Consume configuration bytes
        uint8_t rank;
        if (!consumeBytes(data, size, offset, rank)) return 0;
        rank = (rank % 5) + 1; // Rank between 1 and 5
        
        uint8_t dtype_choice;
        if (!consumeBytes(data, size, offset, dtype_choice)) return 0;
        
        int8_t dim;
        if (!consumeBytes(data, size, offset, dim)) return 0;
        
        // Build tensor shape
        std::vector<int64_t> shape;
        for (int i = 0; i < rank; i++) {
            uint8_t dim_size;
            if (!consumeBytes(data, size, offset, dim_size)) {
                shape.push_back(1);
            } else {
                // Allow 0-sized dimensions for edge cases
                shape.push_back(dim_size % 10); // Keep dimensions small
            }
        }
        
        // Determine dtype
        torch::ScalarType dtype;
        switch (dtype_choice % 4) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            default: dtype = torch::kBFloat16; break;
        }
        
        // Calculate total elements
        int64_t total_elements = 1;
        for (auto s : shape) {
            total_elements *= s;
        }
        
        // Create tensor with random data from fuzzer input
        torch::Tensor input;
        
        if (total_elements == 0) {
            // Handle empty tensor case
            input = torch::empty(shape, torch::dtype(dtype));
        } else if (total_elements > 100000) {
            // Prevent excessive memory usage
            return 0;
        } else {
            // Fill tensor with fuzzer data
            input = torch::empty(shape, torch::dtype(dtype));
            
            if (dtype == torch::kFloat32) {
                float* data_ptr = input.data_ptr<float>();
                for (int64_t i = 0; i < total_elements; i++) {
                    float val = 0.0f;
                    if (offset < size) {
                        val = static_cast<float>((data[offset++] - 128) / 32.0f);
                    }
                    data_ptr[i] = val;
                }
            } else if (dtype == torch::kFloat64) {
                double* data_ptr = input.data_ptr<double>();
                for (int64_t i = 0; i < total_elements; i++) {
                    double val = 0.0;
                    if (offset < size) {
                        val = static_cast<double>((data[offset++] - 128) / 32.0);
                    }
                    data_ptr[i] = val;
                }
            } else {
                // For Float16/BFloat16, convert through Float32
                input = input.to(torch::kFloat32);
                float* data_ptr = input.data_ptr<float>();
                for (int64_t i = 0; i < total_elements; i++) {
                    float val = 0.0f;
                    if (offset < size) {
                        val = static_cast<float>((data[offset++] - 128) / 32.0f);
                    }
                    data_ptr[i] = val;
                }
                input = input.to(dtype);
            }
        }
        
        // Test with special values
        if (offset < size && data[offset++] % 10 == 0) {
            if (total_elements > 0) {
                // Inject special values
                uint8_t special_type;
                if (consumeBytes(data, size, offset, special_type)) {
                    torch::Tensor special = input.clone();
                    switch (special_type % 5) {
                        case 0: special.fill_(std::numeric_limits<float>::infinity()); break;
                        case 1: special.fill_(-std::numeric_limits<float>::infinity()); break;
                        case 2: special.fill_(std::numeric_limits<float>::quiet_NaN()); break;
                        case 3: special.fill_(std::numeric_limits<float>::max()); break;
                        case 4: special.fill_(std::numeric_limits<float>::min()); break;
                    }
                    input = special;
                }
            }
        }
        
        // Create LogSoftmax module
        int64_t actual_dim = dim;
        if (rank > 0) {
            // Normalize dim to valid range
            actual_dim = dim % rank;
            if (actual_dim < 0) actual_dim += rank;
        } else {
            actual_dim = 0;
        }
        
        torch::nn::LogSoftmax log_softmax(torch::nn::LogSoftmaxOptions(actual_dim));
        
        // Apply LogSoftmax
        torch::Tensor output = log_softmax->forward(input);
        
        // Additional operations to increase coverage
        if (offset < size && data[offset++] % 2 == 0) {
            // Test backward pass
            if (output.requires_grad()) {
                torch::Tensor grad_output = torch::ones_like(output);
                output.backward(grad_output);
            }
        }
        
        // Test with different memory layouts
        if (offset < size && data[offset++] % 3 == 0) {
            if (rank >= 2) {
                torch::Tensor transposed = input.transpose(0, rank - 1);
                torch::Tensor output2 = log_softmax->forward(transposed);
            }
        }
        
        // Test with non-contiguous tensors
        if (offset < size && data[offset++] % 3 == 1 && total_elements > 0) {
            torch::Tensor strided = input.as_strided(shape, shape);
            torch::Tensor output3 = log_softmax->forward(strided);
        }
        
        // Test with requires_grad
        if (offset < size && data[offset++] % 2 == 0) {
            input.requires_grad_(true);
            torch::Tensor output4 = log_softmax->forward(input);
            if (total_elements > 0) {
                torch::Tensor loss = output4.sum();
                loss.backward();
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected in fuzzing
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions
        return 0;
    }
    
    return 0;
}