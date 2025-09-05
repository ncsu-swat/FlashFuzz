#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Helper to consume bytes from the fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t& offset, size_t size, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    try {
        if (size < 16) return 0;  // Need minimum bytes for basic params
        
        size_t offset = 0;
        
        // Consume parameters for tensor creation
        uint8_t rank;
        if (!consumeBytes(data, offset, size, rank)) return 0;
        rank = (rank % 5) + 1;  // Limit rank to 1-5
        
        // Build shape
        std::vector<int64_t> shape;
        for (size_t i = 0; i < rank; ++i) {
            uint8_t dim;
            if (!consumeBytes(data, offset, size, dim)) {
                shape.push_back(1);
            } else {
                shape.push_back(static_cast<int64_t>(dim % 10));  // Allow 0-dim
            }
        }
        
        // Consume dtype selector
        uint8_t dtype_selector;
        if (!consumeBytes(data, offset, size, dtype_selector)) dtype_selector = 0;
        
        // Select dtype
        torch::ScalarType dtype;
        switch (dtype_selector % 4) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            case 3: dtype = torch::kBFloat16; break;
            default: dtype = torch::kFloat32;
        }
        
        // Consume device selector
        uint8_t device_selector;
        if (!consumeBytes(data, offset, size, device_selector)) device_selector = 0;
        torch::Device device = (device_selector % 2 == 0) ? torch::kCPU : torch::kCPU;
        
        // Consume requires_grad
        uint8_t requires_grad;
        if (!consumeBytes(data, offset, size, requires_grad)) requires_grad = 0;
        
        // Create tensor options
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .requires_grad(requires_grad % 2 == 1);
        
        // Create input tensor
        torch::Tensor input;
        
        // Consume tensor creation method
        uint8_t creation_method;
        if (!consumeBytes(data, offset, size, creation_method)) creation_method = 0;
        
        switch (creation_method % 5) {
            case 0:
                input = torch::randn(shape, options);
                break;
            case 1:
                input = torch::ones(shape, options);
                break;
            case 2:
                input = torch::zeros(shape, options);
                break;
            case 3:
                input = torch::empty(shape, options);
                break;
            case 4: {
                // Create from remaining data
                size_t elem_count = 1;
                for (auto d : shape) elem_count *= std::max(int64_t(1), d);
                
                if (elem_count > 0 && elem_count < 10000) {
                    std::vector<float> values;
                    for (size_t i = 0; i < elem_count; ++i) {
                        float val;
                        if (consumeBytes(data, offset, size, val)) {
                            values.push_back(val);
                        } else {
                            values.push_back(static_cast<float>(i));
                        }
                    }
                    input = torch::from_blob(values.data(), shape, torch::kFloat32).clone().to(options);
                } else {
                    input = torch::randn(shape, options);
                }
                break;
            }
        }
        
        // Consume parameters for rrelu
        float lower, upper;
        bool training;
        
        if (!consumeBytes(data, offset, size, lower)) lower = 0.125f;
        if (!consumeBytes(data, offset, size, upper)) upper = 0.333f;
        
        // Normalize lower/upper to valid range
        lower = std::abs(lower);
        upper = std::abs(upper);
        if (std::isnan(lower) || std::isinf(lower)) lower = 0.125f;
        if (std::isnan(upper) || std::isinf(upper)) upper = 0.333f;
        
        // Ensure lower <= upper
        if (lower > upper) std::swap(lower, upper);
        
        // Clamp to reasonable range
        lower = std::min(lower, 1.0f);
        upper = std::min(upper, 1.0f);
        
        uint8_t training_byte;
        if (!consumeBytes(data, offset, size, training_byte)) training_byte = 0;
        training = (training_byte % 2 == 1);
        
        // Test different rrelu variants
        uint8_t variant;
        if (!consumeBytes(data, offset, size, variant)) variant = 0;
        
        torch::Tensor result;
        switch (variant % 3) {
            case 0:
                // In-place version
                result = input.clone();
                torch::nn::functional::rrelu(result, 
                    torch::nn::functional::RReLUFuncOptions()
                        .lower(lower)
                        .upper(upper)
                        .training(training)
                        .inplace(true));
                break;
            case 1:
                // Out-of-place version
                result = torch::nn::functional::rrelu(input,
                    torch::nn::functional::RReLUFuncOptions()
                        .lower(lower)
                        .upper(upper)
                        .training(training)
                        .inplace(false));
                break;
            case 2:
                // Using module
                {
                    auto rrelu_module = torch::nn::RReLU(torch::nn::RReLUOptions()
                        .lower(lower)
                        .upper(upper));
                    if (training) {
                        rrelu_module->train();
                    } else {
                        rrelu_module->eval();
                    }
                    result = rrelu_module->forward(input);
                }
                break;
        }
        
        // Perform some operations on result to trigger potential issues
        if (result.numel() > 0) {
            auto sum = result.sum();
            auto mean = result.mean();
            
            // Test backward if requires_grad
            if (result.requires_grad()) {
                try {
                    sum.backward();
                } catch (...) {
                    // Ignore backward errors
                }
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected
        return 0;
    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions
        return 0;
    }
    
    return 0;
}