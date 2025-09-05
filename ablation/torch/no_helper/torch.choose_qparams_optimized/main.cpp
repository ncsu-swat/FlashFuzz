#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Helper to consume bytes from the fuzzer input
template<typename T>
bool consumeBytes(const uint8_t*& data, size_t& size, T& value) {
    if (size < sizeof(T)) return false;
    std::memcpy(&value, data, sizeof(T));
    data += sizeof(T);
    size -= sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        if (size < 16) return 0;  // Need minimum bytes for basic parameters
        
        const uint8_t* ptr = data;
        size_t remaining = size;
        
        // Consume parameters for tensor creation
        uint8_t num_dims;
        if (!consumeBytes(ptr, remaining, num_dims)) return 0;
        num_dims = (num_dims % 5) + 1;  // 1-5 dimensions
        
        // Build shape
        std::vector<int64_t> shape;
        for (int i = 0; i < num_dims; i++) {
            uint16_t dim_size;
            if (!consumeBytes(ptr, remaining, dim_size)) {
                dim_size = 1;
            }
            // Allow 0-sized dimensions for edge cases
            shape.push_back(dim_size % 100);  // Cap at 100 to avoid OOM
        }
        
        // Consume dtype selector
        uint8_t dtype_selector;
        if (!consumeBytes(ptr, remaining, dtype_selector)) {
            dtype_selector = 0;
        }
        
        // Select dtype - choose_qparams_optimized typically works with floating point tensors
        torch::ScalarType dtype;
        switch (dtype_selector % 4) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            case 3: dtype = torch::kBFloat16; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Consume parameters for choose_qparams_optimized
        uint8_t qmin_raw, qmax_raw;
        if (!consumeBytes(ptr, remaining, qmin_raw)) qmin_raw = 0;
        if (!consumeBytes(ptr, remaining, qmax_raw)) qmax_raw = 255;
        
        // Ensure qmin < qmax
        int64_t qmin = qmin_raw;
        int64_t qmax = qmax_raw;
        if (qmin >= qmax) {
            qmax = qmin + 1;
        }
        
        // Consume numel selector for tensor values
        uint8_t numel_selector;
        if (!consumeBytes(ptr, remaining, numel_selector)) {
            numel_selector = 0;
        }
        
        // Create tensor with various strategies
        torch::Tensor tensor;
        
        if (numel_selector % 4 == 0) {
            // Create from shape with random values
            tensor = torch::randn(shape, torch::TensorOptions().dtype(dtype));
        } else if (numel_selector % 4 == 1) {
            // Create empty tensor
            tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));
        } else if (numel_selector % 4 == 2) {
            // Create with specific values from fuzzer data
            int64_t numel = 1;
            for (auto dim : shape) {
                numel *= dim;
            }
            
            if (numel > 0 && numel < 10000) {  // Cap to avoid OOM
                std::vector<float> values;
                for (int64_t i = 0; i < numel; i++) {
                    float val;
                    if (remaining >= sizeof(float)) {
                        consumeBytes(ptr, remaining, val);
                    } else {
                        val = static_cast<float>(i);
                    }
                    values.push_back(val);
                }
                tensor = torch::from_blob(values.data(), shape, torch::kFloat32).clone().to(dtype);
            } else {
                tensor = torch::zeros(shape, torch::TensorOptions().dtype(dtype));
            }
        } else {
            // Create with special values
            uint8_t special_val;
            if (!consumeBytes(ptr, remaining, special_val)) special_val = 0;
            switch (special_val % 5) {
                case 0: tensor = torch::zeros(shape, torch::TensorOptions().dtype(dtype)); break;
                case 1: tensor = torch::ones(shape, torch::TensorOptions().dtype(dtype)); break;
                case 2: tensor = torch::full(shape, std::numeric_limits<float>::infinity(), torch::TensorOptions().dtype(dtype)); break;
                case 3: tensor = torch::full(shape, -std::numeric_limits<float>::infinity(), torch::TensorOptions().dtype(dtype)); break;
                case 4: tensor = torch::full(shape, std::numeric_limits<float>::quiet_NaN(), torch::TensorOptions().dtype(dtype)); break;
            }
        }
        
        // Test with different tensor layouts
        uint8_t layout_selector;
        if (consumeBytes(ptr, remaining, layout_selector)) {
            if (layout_selector % 3 == 1 && tensor.numel() > 0) {
                // Make non-contiguous
                tensor = tensor.transpose(0, -1);
            } else if (layout_selector % 3 == 2 && tensor.dim() >= 2) {
                // Permute dimensions
                std::vector<int64_t> perm;
                for (int64_t i = tensor.dim() - 1; i >= 0; i--) {
                    perm.push_back(i);
                }
                tensor = tensor.permute(perm);
            }
        }
        
        // Call choose_qparams_optimized
        auto result = torch::choose_qparams_optimized(tensor, qmin, qmax);
        
        // Access the results to ensure they're computed
        auto scale = std::get<0>(result);
        auto zero_point = std::get<1>(result);
        
        // Optionally use the results to trigger more code paths
        if (scale.numel() > 0) {
            scale.item<double>();
        }
        if (zero_point.numel() > 0) {
            zero_point.item<int64_t>();
        }
        
        // Test with different bit widths
        uint8_t bit_width;
        if (consumeBytes(ptr, remaining, bit_width)) {
            bit_width = (bit_width % 8) + 1;  // 1-8 bits
            int64_t qmax_bits = (1 << bit_width) - 1;
            
            try {
                auto result2 = torch::choose_qparams_optimized(tensor, 0, qmax_bits);
                std::get<0>(result2);
                std::get<1>(result2);
            } catch (const c10::Error& e) {
                // Quantization errors are expected for some inputs
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch errors are expected for invalid operations
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}