#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consume(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

// Helper to create tensor from fuzzer input
torch::Tensor createTensorFromBytes(const uint8_t* data, size_t size, size_t& offset) {
    // Consume parameters for tensor creation
    uint8_t rank;
    if (!consume(data, size, offset, rank)) {
        return torch::randn({1});
    }
    rank = (rank % 5) + 1; // Limit rank to 1-5 dimensions
    
    std::vector<int64_t> shape;
    for (int i = 0; i < rank; i++) {
        uint8_t dim;
        if (!consume(data, size, offset, dim)) {
            shape.push_back(1);
        } else {
            // Allow 0-sized dimensions for edge cases
            shape.push_back(dim % 32); // Limit dimension size
        }
    }
    
    // Consume dtype choice
    uint8_t dtype_choice;
    if (!consume(data, size, offset, dtype_choice)) {
        dtype_choice = 0;
    }
    
    torch::ScalarType dtype;
    switch (dtype_choice % 4) {
        case 0: dtype = torch::kFloat32; break;
        case 1: dtype = torch::kFloat64; break;
        case 2: dtype = torch::kFloat16; break;
        case 3: dtype = torch::kBFloat16; break;
        default: dtype = torch::kFloat32;
    }
    
    // Create tensor with random data
    auto options = torch::TensorOptions().dtype(dtype);
    torch::Tensor tensor;
    
    try {
        tensor = torch::randn(shape, options);
        
        // Optionally make it non-contiguous
        uint8_t make_noncontig;
        if (consume(data, size, offset, make_noncontig) && (make_noncontig % 4 == 0)) {
            if (tensor.dim() >= 2 && tensor.size(0) > 1 && tensor.size(1) > 1) {
                tensor = tensor.transpose(0, 1);
            }
        }
        
        // Optionally add some special values
        uint8_t special_vals;
        if (consume(data, size, offset, special_vals)) {
            if (special_vals % 5 == 0 && tensor.numel() > 0) {
                tensor.view(-1)[0] = std::numeric_limits<float>::infinity();
            } else if (special_vals % 5 == 1 && tensor.numel() > 0) {
                tensor.view(-1)[0] = -std::numeric_limits<float>::infinity();
            } else if (special_vals % 5 == 2 && tensor.numel() > 0) {
                tensor.view(-1)[0] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    } catch (...) {
        tensor = torch::randn({1}, options);
    }
    
    return tensor;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        if (size < 10) {
            // Need minimum bytes for basic parameters
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input = createTensorFromBytes(data, size, offset);
        
        // Create running_mean tensor (should match certain dimensions of input)
        torch::Tensor running_mean;
        if (input.dim() >= 2) {
            // For batch norm, running_mean should match channel dimension (typically dim 1)
            int64_t channel_size = input.size(1);
            running_mean = createTensorFromBytes(data, size, offset);
            try {
                running_mean = running_mean.reshape({channel_size});
            } catch (...) {
                running_mean = torch::zeros({channel_size}, input.options());
            }
        } else {
            running_mean = torch::zeros({1}, input.options());
        }
        
        // Create running_var tensor (same shape as running_mean)
        torch::Tensor running_var;
        try {
            running_var = createTensorFromBytes(data, size, offset);
            running_var = running_var.reshape(running_mean.sizes());
            // Ensure variance is non-negative for some cases
            uint8_t make_positive;
            if (consume(data, size, offset, make_positive) && (make_positive % 2 == 0)) {
                running_var = running_var.abs();
            }
        } catch (...) {
            running_var = torch::ones_like(running_mean);
        }
        
        // Consume momentum value
        float momentum = 0.1f;
        consume(data, size, offset, momentum);
        // Clamp momentum to reasonable range but allow edge cases
        if (!std::isfinite(momentum)) {
            momentum = 0.1f;
        }
        
        // Try to call batch_norm_update_stats
        try {
            // The function signature is typically:
            // std::tuple<Tensor, Tensor> batch_norm_update_stats(
            //     const Tensor& input,
            //     const Tensor& running_mean,
            //     const Tensor& running_var,
            //     double momentum)
            
            auto result = torch::batch_norm_update_stats(
                input,
                running_mean, 
                running_var,
                static_cast<double>(momentum)
            );
            
            // Access the results to ensure computation happens
            auto mean = std::get<0>(result);
            auto var = std::get<1>(result);
            
            // Trigger computation
            if (mean.defined()) {
                mean.sum();
            }
            if (var.defined()) {
                var.sum();
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid inputs
            // Continue fuzzing
        } catch (const std::exception& e) {
            // Other exceptions might indicate issues
            // But continue fuzzing
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}