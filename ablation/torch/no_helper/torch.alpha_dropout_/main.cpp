#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t& offset, size_t size, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        if (size < 16) return 0;  // Need minimum bytes for basic parameters
        
        size_t offset = 0;
        
        // Consume parameters for alpha_dropout_
        float p = 0.5f;
        consumeBytes(data, offset, size, p);
        // Normalize p to [0, 1] range
        p = std::abs(p);
        while (p > 1.0f) p /= 10.0f;
        
        bool train = true;
        if (offset < size) {
            train = (data[offset++] & 1);
        }
        
        // Determine tensor properties
        uint8_t rank = 1;
        if (offset < size) {
            rank = (data[offset++] % 5) + 1;  // 1-5 dimensions
        }
        
        // Build shape
        std::vector<int64_t> shape;
        for (uint8_t i = 0; i < rank; ++i) {
            uint16_t dim = 1;
            if (consumeBytes(data, offset, size, dim)) {
                dim = (dim % 100) + 1;  // Keep dimensions reasonable (1-100)
                shape.push_back(dim);
            } else {
                shape.push_back(1);
            }
        }
        
        // Determine dtype
        uint8_t dtype_selector = 0;
        if (offset < size) {
            dtype_selector = data[offset++] % 4;
        }
        
        torch::ScalarType dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            default: dtype = torch::kBFloat16; break;
        }
        
        // Determine device
        bool use_cuda = false;
        if (offset < size && torch::cuda::is_available()) {
            use_cuda = (data[offset++] & 1);
        }
        
        // Create tensor options
        auto options = torch::TensorOptions().dtype(dtype);
        if (use_cuda) {
            options = options.device(torch::kCUDA);
        }
        
        // Create tensor with various initialization methods
        torch::Tensor tensor;
        uint8_t init_method = 0;
        if (offset < size) {
            init_method = data[offset++] % 6;
        }
        
        switch (init_method) {
            case 0:
                tensor = torch::randn(shape, options);
                break;
            case 1:
                tensor = torch::ones(shape, options);
                break;
            case 2:
                tensor = torch::zeros(shape, options);
                break;
            case 3:
                tensor = torch::rand(shape, options);
                break;
            case 4:
                tensor = torch::empty(shape, options);
                break;
            default:
                tensor = torch::full(shape, 0.5, options);
                break;
        }
        
        // Test with requires_grad
        if (offset < size && (data[offset++] & 1)) {
            if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                tensor.requires_grad_(true);
            }
        }
        
        // Apply alpha_dropout_ in-place
        torch::nn::functional::alpha_dropout(tensor, 
            torch::nn::functional::AlphaDropoutFuncOptions()
                .p(p)
                .training(train)
                .inplace(true));
        
        // Additional edge cases
        if (offset < size) {
            uint8_t edge_case = data[offset++] % 5;
            switch (edge_case) {
                case 0: {
                    // Empty tensor
                    auto empty_tensor = torch::empty({0}, options);
                    torch::nn::functional::alpha_dropout(empty_tensor,
                        torch::nn::functional::AlphaDropoutFuncOptions()
                            .p(p)
                            .training(train)
                            .inplace(true));
                    break;
                }
                case 1: {
                    // Scalar tensor
                    auto scalar_tensor = torch::tensor(1.0, options);
                    torch::nn::functional::alpha_dropout(scalar_tensor,
                        torch::nn::functional::AlphaDropoutFuncOptions()
                            .p(p)
                            .training(train)
                            .inplace(true));
                    break;
                }
                case 2: {
                    // Large tensor
                    if (shape.size() > 0) {
                        shape[0] = 1000;
                        auto large_tensor = torch::randn(shape, options);
                        torch::nn::functional::alpha_dropout(large_tensor,
                            torch::nn::functional::AlphaDropoutFuncOptions()
                                .p(p)
                                .training(train)
                                .inplace(true));
                    }
                    break;
                }
                case 3: {
                    // Test with p=0 and p=1
                    auto test_tensor = torch::randn(shape, options);
                    torch::nn::functional::alpha_dropout(test_tensor,
                        torch::nn::functional::AlphaDropoutFuncOptions()
                            .p(0.0)
                            .training(train)
                            .inplace(true));
                    
                    test_tensor = torch::randn(shape, options);
                    torch::nn::functional::alpha_dropout(test_tensor,
                        torch::nn::functional::AlphaDropoutFuncOptions()
                            .p(1.0)
                            .training(train)
                            .inplace(true));
                    break;
                }
                case 4: {
                    // Non-contiguous tensor
                    if (shape.size() >= 2 && shape[0] > 1 && shape[1] > 1) {
                        auto base = torch::randn({shape[0] * 2, shape[1]}, options);
                        auto strided = base.slice(0, 0, shape[0] * 2, 2);
                        torch::nn::functional::alpha_dropout(strided,
                            torch::nn::functional::AlphaDropoutFuncOptions()
                                .p(p)
                                .training(train)
                                .inplace(true));
                    }
                    break;
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}