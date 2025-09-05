#include <torch/torch.h>
#include <iostream>
#include <vector>
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

// Helper to create tensor from fuzzer input
torch::Tensor createTensorFromBytes(const uint8_t*& data, size_t& size) {
    // Consume parameters for tensor creation
    uint8_t num_dims = 0;
    if (!consumeBytes(data, size, num_dims)) {
        return torch::empty({});
    }
    num_dims = (num_dims % 5) + 1; // Limit to 1-5 dimensions
    
    std::vector<int64_t> shape;
    for (int i = 0; i < num_dims; ++i) {
        uint8_t dim_size = 0;
        if (!consumeBytes(data, size, dim_size)) {
            shape.push_back(1);
        } else {
            // Allow 0-sized dimensions for edge cases
            shape.push_back(dim_size % 10); // Limit individual dimension size
        }
    }
    
    // Determine dtype
    uint8_t dtype_selector = 0;
    if (consumeBytes(data, size, dtype_selector)) {
        dtype_selector = dtype_selector % 3;
    }
    
    torch::ScalarType dtype;
    switch (dtype_selector) {
        case 0: dtype = torch::kFloat32; break;
        case 1: dtype = torch::kFloat64; break;
        case 2: dtype = torch::kFloat16; break;
        default: dtype = torch::kFloat32;
    }
    
    // Create tensor with random data
    auto options = torch::TensorOptions().dtype(dtype);
    torch::Tensor tensor;
    
    try {
        tensor = torch::randn(shape, options);
        
        // Optionally make it non-contiguous
        uint8_t make_non_contiguous = 0;
        if (consumeBytes(data, size, make_non_contiguous) && (make_non_contiguous % 4 == 0)) {
            if (tensor.numel() > 1 && tensor.dim() > 1) {
                tensor = tensor.transpose(0, tensor.dim() - 1);
            }
        }
        
        // Fill with fuzzer data if available
        if (tensor.numel() > 0 && size > 0) {
            size_t bytes_to_copy = std::min(size, (size_t)(tensor.numel() * tensor.element_size()));
            if (bytes_to_copy > 0) {
                std::memcpy(tensor.data_ptr(), data, bytes_to_copy);
                data += bytes_to_copy;
                size -= bytes_to_copy;
            }
        }
    } catch (...) {
        // If tensor creation fails, return empty tensor
        tensor = torch::empty({});
    }
    
    return tensor;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0; // Need minimum bytes for basic setup
    
    try {
        // Create input tensor
        torch::Tensor input = createTensorFromBytes(data, size);
        
        // Determine weight tensor configuration
        uint8_t weight_type = 0;
        if (size > 0) {
            consumeBytes(data, size, weight_type);
            weight_type = weight_type % 4;
        }
        
        torch::Tensor weight;
        
        switch (weight_type) {
            case 0: {
                // Scalar weight
                float scalar_val = 0.25f;
                if (size >= sizeof(float)) {
                    consumeBytes(data, size, scalar_val);
                }
                weight = torch::tensor(scalar_val);
                break;
            }
            case 1: {
                // 1-D weight matching channels
                int64_t num_channels = 1;
                if (input.dim() >= 2) {
                    num_channels = input.size(1);
                }
                weight = torch::randn({num_channels});
                if (size > 0 && weight.numel() > 0) {
                    size_t bytes_to_copy = std::min(size, (size_t)(weight.numel() * weight.element_size()));
                    std::memcpy(weight.data_ptr(), data, bytes_to_copy);
                    data += bytes_to_copy;
                    size -= bytes_to_copy;
                }
                break;
            }
            case 2: {
                // 1-D weight with mismatched size (edge case)
                uint8_t weight_size = 1;
                if (size > 0) {
                    consumeBytes(data, size, weight_size);
                    weight_size = (weight_size % 10) + 1;
                }
                weight = torch::randn({weight_size});
                break;
            }
            case 3: {
                // Create weight from remaining bytes
                weight = createTensorFromBytes(data, size);
                break;
            }
            default:
                weight = torch::tensor(0.1f);
        }
        
        // Test various input/weight dtype combinations
        uint8_t dtype_combo = 0;
        if (size > 0) {
            consumeBytes(data, size, dtype_combo);
            dtype_combo = dtype_combo % 4;
        }
        
        switch (dtype_combo) {
            case 1:
                if (input.defined() && input.numel() > 0) {
                    input = input.to(torch::kFloat64);
                }
                if (weight.defined() && weight.numel() > 0) {
                    weight = weight.to(torch::kFloat64);
                }
                break;
            case 2:
                if (input.defined() && input.numel() > 0) {
                    input = input.to(torch::kFloat16);
                }
                if (weight.defined() && weight.numel() > 0) {
                    weight = weight.to(torch::kFloat16);
                }
                break;
            case 3:
                // Mixed dtypes for edge case testing
                if (input.defined() && input.numel() > 0) {
                    input = input.to(torch::kFloat32);
                }
                if (weight.defined() && weight.numel() > 0) {
                    weight = weight.to(torch::kFloat64);
                }
                break;
        }
        
        // Apply PReLU operation
        torch::Tensor result = torch::prelu(input, weight);
        
        // Perform some operations on result to ensure it's valid
        if (result.defined() && result.numel() > 0) {
            // Access the result to ensure computation completes
            float sum = result.sum().item<float>();
            (void)sum; // Avoid unused variable warning
            
            // Test gradient computation if possible
            if (input.requires_grad() || weight.requires_grad()) {
                uint8_t compute_grad = 0;
                if (size > 0 && consumeBytes(data, size, compute_grad) && (compute_grad % 4 == 0)) {
                    try {
                        result.backward(torch::ones_like(result));
                    } catch (...) {
                        // Gradient computation might fail for some configurations
                    }
                }
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid configurations
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