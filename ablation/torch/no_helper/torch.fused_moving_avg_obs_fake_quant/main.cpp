#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <vector>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t*& data, size_t& size, T& out) {
    if (size < sizeof(T)) return false;
    std::memcpy(&out, data, sizeof(T));
    data += sizeof(T);
    size -= sizeof(T);
    return true;
}

// Helper to create tensor from fuzzer input
torch::Tensor createTensorFromBytes(const uint8_t*& data, size_t& size) {
    // Consume parameters for tensor creation
    uint8_t rank;
    if (!consumeBytes(data, size, rank)) {
        return torch::randn({1});
    }
    rank = (rank % 5) + 1; // Limit rank to 1-5
    
    std::vector<int64_t> shape;
    for (uint8_t i = 0; i < rank; ++i) {
        uint8_t dim;
        if (!consumeBytes(data, size, dim)) {
            shape.push_back(1);
        } else {
            // Allow 0-dim tensors for edge cases, but cap at reasonable size
            shape.push_back(dim % 16);
        }
    }
    
    // Consume dtype selector
    uint8_t dtype_selector;
    if (!consumeBytes(data, size, dtype_selector)) {
        dtype_selector = 0;
    }
    
    // Create tensor with appropriate dtype
    torch::Tensor tensor;
    switch (dtype_selector % 4) {
        case 0:
            tensor = torch::randn(shape, torch::kFloat32);
            break;
        case 1:
            tensor = torch::randn(shape, torch::kFloat64);
            break;
        case 2:
            tensor = torch::randn(shape, torch::kFloat16);
            break;
        case 3:
            tensor = torch::randn(shape, torch::kBFloat16);
            break;
    }
    
    // Optionally make tensor non-contiguous
    uint8_t make_noncontig;
    if (consumeBytes(data, size, make_noncontig) && (make_noncontig % 3 == 0) && shape.size() > 1) {
        tensor = tensor.transpose(0, shape.size() - 1);
    }
    
    // Fill with fuzzer data if available
    if (size > 0) {
        size_t elem_size = tensor.element_size();
        size_t total_elems = tensor.numel();
        size_t bytes_needed = elem_size * total_elems;
        
        if (bytes_needed > 0 && size >= bytes_needed) {
            std::memcpy(tensor.data_ptr(), data, bytes_needed);
            data += bytes_needed;
            size -= bytes_needed;
        }
    }
    
    return tensor;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        if (size < 16) return 0; // Need minimum bytes for basic parameters
        
        const uint8_t* ptr = data;
        size_t remaining = size;
        
        // Create input tensor
        torch::Tensor input = createTensorFromBytes(ptr, remaining);
        
        // Create observer tensor (should match input shape or be broadcastable)
        torch::Tensor observer = createTensorFromBytes(ptr, remaining);
        
        // Consume averaging constant
        float averaging_const = 0.01f;
        consumeBytes(ptr, remaining, averaging_const);
        // Clamp to reasonable range
        averaging_const = std::abs(averaging_const);
        if (averaging_const > 1.0f) averaging_const = 1.0f / averaging_const;
        if (averaging_const < 1e-6f) averaging_const = 1e-6f;
        
        // Consume scale
        float scale = 1.0f;
        consumeBytes(ptr, remaining, scale);
        if (std::isnan(scale) || std::isinf(scale)) scale = 1.0f;
        
        // Consume zero_point
        int32_t zero_point = 0;
        consumeBytes(ptr, remaining, zero_point);
        zero_point = zero_point % 256; // Keep in reasonable range
        
        // Consume quant_min and quant_max
        int32_t quant_min = -128;
        int32_t quant_max = 127;
        consumeBytes(ptr, remaining, quant_min);
        consumeBytes(ptr, remaining, quant_max);
        
        // Ensure quant_min < quant_max
        if (quant_min >= quant_max) {
            int32_t temp = quant_min;
            quant_min = quant_max - 1;
            quant_max = temp + 1;
        }
        
        // Consume ch_axis
        int32_t ch_axis = 0;
        consumeBytes(ptr, remaining, ch_axis);
        
        // Consume per_row_fake_quant flag
        bool per_row_fake_quant = false;
        uint8_t flag;
        if (consumeBytes(ptr, remaining, flag)) {
            per_row_fake_quant = (flag % 2) == 1;
        }
        
        // Consume symmetric_quant flag
        bool symmetric_quant = false;
        if (consumeBytes(ptr, remaining, flag)) {
            symmetric_quant = (flag % 2) == 1;
        }
        
        // Try to call the operation
        try {
            // The function signature based on PyTorch source:
            // fused_moving_avg_obs_fake_quant(input, observer, averaging_const, 
            //                                 scale, zero_point, quant_min, quant_max,
            //                                 ch_axis, per_row_fake_quant, symmetric_quant)
            
            auto result = torch::fused_moving_avg_obs_fake_quant(
                input,
                observer,
                averaging_const,
                scale,
                zero_point,
                quant_min,
                quant_max,
                ch_axis,
                per_row_fake_quant,
                symmetric_quant
            );
            
            // Access result to ensure computation happens
            if (result.defined()) {
                result.sum().item<float>();
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid inputs
            // Continue fuzzing
        } catch (const std::runtime_error& e) {
            // Runtime errors might occur with invalid parameters
            // Continue fuzzing
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}