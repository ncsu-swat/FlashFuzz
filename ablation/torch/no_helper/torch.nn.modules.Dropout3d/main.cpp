#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from the fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t& offset, size_t size, T& value) {
    if (offset + sizeof(T) > size) {
        return false;
    }
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 16) {
        return 0;  // Need minimum bytes for basic parameters
    }

    size_t offset = 0;
    
    try {
        // Consume parameters for Dropout3d
        float p_raw;
        uint8_t inplace_flag;
        uint8_t tensor_rank;  // 4D or 5D
        uint8_t dtype_selector;
        
        if (!consumeBytes(data, offset, size, p_raw)) return 0;
        if (!consumeBytes(data, offset, size, inplace_flag)) return 0;
        if (!consumeBytes(data, offset, size, tensor_rank)) return 0;
        if (!consumeBytes(data, offset, size, dtype_selector)) return 0;
        
        // Normalize probability to [0, 1]
        float p = std::abs(p_raw);
        p = p - std::floor(p);  // Get fractional part
        
        bool inplace = (inplace_flag % 2) == 1;
        bool use_5d = (tensor_rank % 2) == 0;
        
        // Determine dtype
        torch::ScalarType dtype;
        switch (dtype_selector % 4) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            default: dtype = torch::kBFloat16; break;
        }
        
        // Consume dimensions
        std::vector<int64_t> dims;
        if (use_5d) {
            // 5D tensor: (N, C, D, H, W)
            for (int i = 0; i < 5; i++) {
                uint16_t dim;
                if (!consumeBytes(data, offset, size, dim)) {
                    dim = 1;  // Default dimension
                }
                dims.push_back(static_cast<int64_t>((dim % 128) + 1));  // Limit size for memory
            }
        } else {
            // 4D tensor: (C, D, H, W)
            for (int i = 0; i < 4; i++) {
                uint16_t dim;
                if (!consumeBytes(data, offset, size, dim)) {
                    dim = 1;  // Default dimension
                }
                dims.push_back(static_cast<int64_t>((dim % 128) + 1));  // Limit size for memory
            }
        }
        
        // Create Dropout3d module
        torch::nn::Dropout3d dropout3d(torch::nn::Dropout3dOptions(p).inplace(inplace));
        
        // Create input tensor with various configurations
        torch::Tensor input;
        
        // Use remaining bytes to determine tensor creation method
        if (offset < size) {
            uint8_t tensor_creation_method = data[offset++];
            
            switch (tensor_creation_method % 6) {
                case 0:
                    input = torch::randn(dims, torch::dtype(dtype));
                    break;
                case 1:
                    input = torch::ones(dims, torch::dtype(dtype));
                    break;
                case 2:
                    input = torch::zeros(dims, torch::dtype(dtype));
                    break;
                case 3:
                    input = torch::rand(dims, torch::dtype(dtype));
                    break;
                case 4:
                    // Create with specific values from fuzzer data
                    {
                        int64_t total_elements = 1;
                        for (auto d : dims) total_elements *= d;
                        if (total_elements > 100000) total_elements = 100000;  // Limit for memory
                        
                        std::vector<float> values;
                        for (int64_t i = 0; i < total_elements && offset < size; i++) {
                            uint8_t val;
                            consumeBytes(data, offset, size, val);
                            values.push_back(static_cast<float>(val) / 255.0f);
                        }
                        while (values.size() < total_elements) {
                            values.push_back(0.5f);
                        }
                        
                        auto temp = torch::from_blob(values.data(), {total_elements}, torch::kFloat32);
                        input = temp.reshape(dims).to(dtype);
                    }
                    break;
                case 5:
                    // Edge case: empty tensor (if dimensions allow)
                    if (dims.size() >= 2) {
                        dims[1] = 0;  // Zero channels
                    }
                    input = torch::empty(dims, torch::dtype(dtype));
                    break;
                default:
                    input = torch::randn(dims, torch::dtype(dtype));
            }
        } else {
            input = torch::randn(dims, torch::dtype(dtype));
        }
        
        // Test with different training modes
        if (offset < size) {
            uint8_t training_mode = data[offset++];
            dropout3d->train(training_mode % 2 == 0);
        }
        
        // Apply dropout
        torch::Tensor output = dropout3d->forward(input);
        
        // Perform various operations to increase coverage
        if (offset < size) {
            uint8_t post_op = data[offset++];
            switch (post_op % 8) {
                case 0:
                    output.sum().backward();
                    break;
                case 1:
                    output.mean();
                    break;
                case 2:
                    output.max();
                    break;
                case 3:
                    output.min();
                    break;
                case 4:
                    if (output.numel() > 0) {
                        output.view(-1);
                    }
                    break;
                case 5:
                    output.contiguous();
                    break;
                case 6:
                    // Test with requires_grad
                    if (!inplace && input.defined() && input.numel() > 0) {
                        input.requires_grad_(true);
                        auto out2 = dropout3d->forward(input);
                        if (out2.requires_grad()) {
                            out2.sum().backward();
                        }
                    }
                    break;
                case 7:
                    // Test multiple forward passes
                    for (int i = 0; i < 3; i++) {
                        dropout3d->forward(input);
                    }
                    break;
            }
        }
        
        // Test edge cases with different p values
        if (offset < size) {
            uint8_t edge_case = data[offset++];
            switch (edge_case % 4) {
                case 0:
                    dropout3d = torch::nn::Dropout3d(torch::nn::Dropout3dOptions(0.0).inplace(false));
                    dropout3d->forward(input);
                    break;
                case 1:
                    dropout3d = torch::nn::Dropout3d(torch::nn::Dropout3dOptions(1.0).inplace(false));
                    dropout3d->forward(input);
                    break;
                case 2:
                    dropout3d = torch::nn::Dropout3d(torch::nn::Dropout3dOptions(0.5).inplace(false));
                    dropout3d->eval();
                    dropout3d->forward(input);
                    break;
                case 3:
                    // Test with very small tensor
                    {
                        auto small_input = torch::randn({1, 1, 1, 1, 1}, torch::dtype(dtype));
                        dropout3d->forward(small_input);
                    }
                    break;
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific exceptions are expected for invalid operations
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