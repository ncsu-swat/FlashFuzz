#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Helper function to consume bytes from the fuzzer input
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
    try {
        if (size < 4) {
            return 0;
        }

        size_t offset = 0;
        
        // Consume configuration bytes
        uint8_t rank;
        if (!consumeBytes(data, offset, size, rank)) {
            return 0;
        }
        rank = (rank % 5) + 1; // Limit rank to 1-5 dimensions
        
        uint8_t dtype_selector;
        if (!consumeBytes(data, offset, size, dtype_selector)) {
            return 0;
        }
        
        uint8_t device_selector;
        if (!consumeBytes(data, offset, size, device_selector)) {
            return 0;
        }
        
        uint8_t requires_grad;
        if (!consumeBytes(data, offset, size, requires_grad)) {
            return 0;
        }
        
        // Build shape vector
        std::vector<int64_t> shape;
        for (size_t i = 0; i < rank; ++i) {
            uint8_t dim_size;
            if (!consumeBytes(data, offset, size, dim_size)) {
                shape.push_back(1);
            } else {
                // Allow 0-sized dimensions and various sizes
                shape.push_back(dim_size % 10);
            }
        }
        
        // Select dtype
        torch::ScalarType dtype;
        switch (dtype_selector % 4) {
            case 0:
                dtype = torch::kFloat32;
                break;
            case 1:
                dtype = torch::kFloat64;
                break;
            case 2:
                dtype = torch::kFloat16;
                break;
            case 3:
                dtype = torch::kBFloat16;
                break;
            default:
                dtype = torch::kFloat32;
        }
        
        // Select device
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available() && (device_selector % 2 == 1)) {
            device = torch::Device(torch::kCUDA);
        }
        
        // Create tensor options
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .requires_grad((requires_grad % 2) == 1);
        
        // Create tensor with various initialization strategies
        torch::Tensor tensor;
        uint8_t init_strategy;
        if (consumeBytes(data, offset, size, init_strategy)) {
            switch (init_strategy % 6) {
                case 0:
                    tensor = torch::zeros(shape, options);
                    break;
                case 1:
                    tensor = torch::ones(shape, options);
                    break;
                case 2:
                    tensor = torch::randn(shape, options);
                    break;
                case 3:
                    tensor = torch::rand(shape, options);
                    break;
                case 4:
                    // Fill with specific values from fuzzer data
                    tensor = torch::empty(shape, options);
                    if (tensor.numel() > 0 && offset < size) {
                        size_t elements_to_fill = std::min(
                            static_cast<size_t>(tensor.numel()),
                            (size - offset) / sizeof(float)
                        );
                        std::vector<float> values;
                        for (size_t i = 0; i < elements_to_fill; ++i) {
                            float val;
                            if (consumeBytes(data, offset, size, val)) {
                                values.push_back(val);
                            } else {
                                values.push_back(0.0f);
                            }
                        }
                        if (!values.empty()) {
                            auto flat_tensor = tensor.flatten();
                            for (size_t i = 0; i < values.size() && i < flat_tensor.numel(); ++i) {
                                flat_tensor[i] = values[i];
                            }
                        }
                    }
                    break;
                case 5:
                    // Create with special values
                    tensor = torch::full(shape, std::numeric_limits<float>::infinity(), options);
                    break;
                default:
                    tensor = torch::zeros(shape, options);
            }
        } else {
            tensor = torch::randn(shape, options);
        }
        
        // Test with strided tensors
        uint8_t make_strided;
        if (consumeBytes(data, offset, size, make_strided) && (make_strided % 3 == 0)) {
            if (tensor.dim() >= 2) {
                tensor = tensor.transpose(0, tensor.dim() - 1);
            }
        }
        
        // Test with non-contiguous tensors
        uint8_t make_noncontiguous;
        if (consumeBytes(data, offset, size, make_noncontiguous) && (make_noncontiguous % 3 == 0)) {
            if (tensor.numel() > 1) {
                tensor = tensor.narrow(0, 0, std::max(int64_t(1), tensor.size(0) / 2));
            }
        }
        
        // Apply erfc_ (in-place operation)
        tensor.erfc_();
        
        // Optionally perform additional operations to stress test
        uint8_t extra_ops;
        if (consumeBytes(data, offset, size, extra_ops)) {
            switch (extra_ops % 4) {
                case 0:
                    // Chain another erfc_
                    tensor.erfc_();
                    break;
                case 1:
                    // Access some values
                    if (tensor.numel() > 0) {
                        auto item = tensor.flatten()[0].item<float>();
                        (void)item;
                    }
                    break;
                case 2:
                    // Check properties
                    tensor.is_contiguous();
                    tensor.is_cuda();
                    tensor.requires_grad();
                    break;
                case 3:
                    // Clone and apply again
                    auto cloned = tensor.clone();
                    cloned.erfc_();
                    break;
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
        return -1;
    }
    
    return 0;
}