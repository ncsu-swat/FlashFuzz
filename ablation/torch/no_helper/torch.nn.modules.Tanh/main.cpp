#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from the fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) {
        return false;
    }
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) {
        return 0;
    }

    try {
        size_t offset = 0;
        
        // Consume configuration bytes
        uint8_t num_dims = 0;
        uint8_t dtype_selector = 0;
        uint8_t requires_grad = 0;
        uint8_t use_cuda = 0;
        
        if (!consumeBytes(data, size, offset, num_dims)) return 0;
        if (!consumeBytes(data, size, offset, dtype_selector)) return 0;
        if (!consumeBytes(data, size, offset, requires_grad)) return 0;
        if (!consumeBytes(data, size, offset, use_cuda)) return 0;
        
        // Limit dimensions to reasonable range
        num_dims = num_dims % 6;  // 0-5 dimensions
        
        // Create shape vector
        std::vector<int64_t> shape;
        for (int i = 0; i < num_dims; ++i) {
            uint8_t dim_size = 0;
            if (!consumeBytes(data, size, offset, dim_size)) {
                dim_size = 1;
            }
            // Allow 0-sized dimensions for edge cases
            shape.push_back(dim_size % 16);  // Limit each dimension to 0-15
        }
        
        // Select dtype based on fuzzer input
        torch::ScalarType dtype;
        switch (dtype_selector % 6) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            case 3: dtype = torch::kBFloat16; break;
            case 4: dtype = torch::kInt32; break;
            case 5: dtype = torch::kInt64; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Create tensor options
        auto options = torch::TensorOptions().dtype(dtype);
        
        // Use CUDA if available and requested
        if (use_cuda % 2 == 1 && torch::cuda::is_available()) {
            options = options.device(torch::kCUDA);
        }
        
        // Set requires_grad for floating point types
        bool grad_enabled = (requires_grad % 2 == 1) && 
                           (dtype == torch::kFloat32 || dtype == torch::kFloat64 || 
                            dtype == torch::kFloat16 || dtype == torch::kBFloat16);
        
        // Create input tensor
        torch::Tensor input;
        
        // Handle empty tensor case
        if (shape.empty()) {
            // Scalar tensor
            float scalar_val = 0.0f;
            consumeBytes(data, size, offset, scalar_val);
            input = torch::tensor(scalar_val, options);
        } else {
            // Multi-dimensional tensor
            int64_t total_elements = 1;
            for (auto dim : shape) {
                total_elements *= dim;
            }
            
            if (total_elements == 0) {
                // Empty tensor with shape
                input = torch::empty(shape, options);
            } else if (total_elements > 0 && total_elements <= 1000) {
                // Fill tensor with fuzzer data
                input = torch::empty(shape, options);
                
                // Fill with random values from remaining fuzzer data
                if (offset < size) {
                    uint8_t fill_type = data[offset++] % 5;
                    switch (fill_type) {
                        case 0:
                            input.fill_(0.0);
                            break;
                        case 1:
                            input.fill_(1.0);
                            break;
                        case 2:
                            input.fill_(-1.0);
                            break;
                        case 3:
                            input.uniform_(-10.0, 10.0);
                            break;
                        case 4:
                            // Use remaining bytes as values
                            if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                                float val = 0.0f;
                                consumeBytes(data, size, offset, val);
                                input.fill_(val);
                            } else {
                                input.random_(-100, 100);
                            }
                            break;
                    }
                } else {
                    input.uniform_(-1.0, 1.0);
                }
            } else {
                // Too large, create smaller tensor
                input = torch::randn({2, 3}, options);
            }
        }
        
        if (grad_enabled) {
            input.requires_grad_(true);
        }
        
        // Create and apply Tanh module
        torch::nn::Tanh tanh_module;
        torch::Tensor output = tanh_module(input);
        
        // Exercise additional functionality
        if (grad_enabled && output.requires_grad()) {
            // Compute gradient
            torch::Tensor grad_output = torch::ones_like(output);
            output.backward(grad_output);
            
            // Access gradient if available
            if (input.grad().defined()) {
                torch::Tensor grad = input.grad();
                // Force computation
                grad.sum();
            }
        }
        
        // Force computation and verify output properties
        output.sum();
        
        // Verify shape preservation
        if (input.sizes() != output.sizes()) {
            std::cerr << "Shape mismatch!" << std::endl;
        }
        
        // Test edge cases with special values
        if (offset + 1 < size && data[offset] % 4 == 0) {
            torch::Tensor special_input;
            uint8_t special_case = data[offset + 1] % 5;
            
            switch (special_case) {
                case 0:
                    special_input = torch::tensor({std::numeric_limits<float>::infinity()}, options);
                    break;
                case 1:
                    special_input = torch::tensor({-std::numeric_limits<float>::infinity()}, options);
                    break;
                case 2:
                    special_input = torch::tensor({std::numeric_limits<float>::quiet_NaN()}, options);
                    break;
                case 3:
                    special_input = torch::tensor({0.0f}, options);
                    break;
                case 4:
                    special_input = torch::tensor({std::numeric_limits<float>::min()}, options);
                    break;
            }
            
            torch::Tensor special_output = tanh_module(special_input);
            special_output.sum();
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for some inputs
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}