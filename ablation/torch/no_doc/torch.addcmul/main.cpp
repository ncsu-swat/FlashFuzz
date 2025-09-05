#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

// Helper to consume a scalar value from fuzzer data
template<typename T>
T consumeScalar(const uint8_t* data, size_t& offset, size_t size) {
    T value = T(1);  // Default value
    if (offset + sizeof(T) <= size) {
        std::memcpy(&value, data + offset, sizeof(T));
        offset += sizeof(T);
    }
    return value;
}

// Helper to create tensor with specific memory layout options
torch::Tensor createTensorWithOptions(const uint8_t* data, size_t& offset, size_t size, bool allow_non_contiguous) {
    torch::Tensor tensor = fuzzer_utils::createTensor(data, size, offset);
    
    if (allow_non_contiguous && offset < size) {
        uint8_t layout_choice = data[offset++];
        
        // Apply various tensor transformations to create different memory layouts
        switch (layout_choice % 8) {
            case 0:
                // Already contiguous
                break;
            case 1:
                // Transpose if possible (2D or higher)
                if (tensor.dim() >= 2) {
                    tensor = tensor.transpose(0, 1);
                }
                break;
            case 2:
                // Permute dimensions if 3D or higher
                if (tensor.dim() >= 3) {
                    std::vector<int64_t> dims;
                    for (int64_t i = tensor.dim() - 1; i >= 0; --i) {
                        dims.push_back(i);
                    }
                    tensor = tensor.permute(dims);
                }
                break;
            case 3:
                // Create a view with different strides
                if (tensor.numel() > 0) {
                    tensor = tensor.as_strided(tensor.sizes(), tensor.strides());
                }
                break;
            case 4:
                // Squeeze and unsqueeze to change layout
                if (tensor.dim() > 0) {
                    tensor = tensor.unsqueeze(0).squeeze(0);
                }
                break;
            case 5:
                // Narrow the tensor if possible
                if (tensor.dim() > 0 && tensor.size(0) > 1) {
                    tensor = tensor.narrow(0, 0, tensor.size(0) / 2 + 1);
                }
                break;
            case 6:
                // Expand tensor with broadcasting
                if (tensor.dim() > 0) {
                    auto sizes = tensor.sizes().vec();
                    for (auto& s : sizes) {
                        if (s == 1 && offset < size) {
                            s = 1 + (data[offset++] % 4);
                        }
                    }
                    tensor = tensor.expand(sizes);
                }
                break;
            case 7:
                // Create a slice
                if (tensor.dim() > 0 && tensor.size(0) > 2) {
                    tensor = tensor.slice(0, 1, tensor.size(0));
                }
                break;
        }
    }
    
    return tensor;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) {  // Minimum size for basic operation
        return 0;
    }
    
    try {
        size_t offset = 0;
        
        // Control flags from fuzzer input
        uint8_t flags = data[offset++];
        bool use_out_tensor = (flags & 0x01) != 0;
        bool allow_non_contiguous = (flags & 0x02) != 0;
        bool use_complex_scalar = (flags & 0x04) != 0;
        bool force_broadcasting = (flags & 0x08) != 0;
        bool use_inplace = (flags & 0x10) != 0;
        
        // Create input tensor
        torch::Tensor input = createTensorWithOptions(data, offset, size, allow_non_contiguous);
        
        // Create tensor1
        torch::Tensor tensor1;
        if (force_broadcasting && offset < size) {
            // Create tensor1 with potentially different shape for broadcasting
            uint8_t broadcast_type = data[offset++];
            tensor1 = createTensorWithOptions(data, offset, size, allow_non_contiguous);
            
            // Modify shape for broadcasting scenarios
            if (broadcast_type % 4 == 0 && tensor1.dim() > 0) {
                // Make it a scalar tensor
                tensor1 = tensor1.reshape({});
            } else if (broadcast_type % 4 == 1 && tensor1.dim() > 1) {
                // Add singleton dimensions
                tensor1 = tensor1.unsqueeze(0);
            } else if (broadcast_type % 4 == 2 && input.dim() > 0) {
                // Try to match some dimensions
                auto target_dim = input.dim();
                while (tensor1.dim() < target_dim) {
                    tensor1 = tensor1.unsqueeze(0);
                }
            }
        } else {
            tensor1 = createTensorWithOptions(data, offset, size, allow_non_contiguous);
        }
        
        // Create tensor2
        torch::Tensor tensor2 = createTensorWithOptions(data, offset, size, allow_non_contiguous);
        
        // Parse scalar value
        torch::Scalar value;
        if (use_complex_scalar && offset + sizeof(std::complex<double>) <= size) {
            double real = consumeScalar<double>(data, offset, size);
            double imag = consumeScalar<double>(data, offset, size);
            value = torch::Scalar(std::complex<double>(real, imag));
        } else {
            double scalar_val = consumeScalar<double>(data, offset, size);
            // Test edge cases for scalar
            if (offset < size) {
                uint8_t scalar_type = data[offset++];
                switch (scalar_type % 6) {
                    case 0: value = torch::Scalar(scalar_val); break;
                    case 1: value = torch::Scalar(0.0); break;
                    case 2: value = torch::Scalar(1.0); break;
                    case 3: value = torch::Scalar(-1.0); break;
                    case 4: value = torch::Scalar(std::numeric_limits<double>::infinity()); break;
                    case 5: value = torch::Scalar(std::numeric_limits<double>::quiet_NaN()); break;
                }
            } else {
                value = torch::Scalar(scalar_val);
            }
        }
        
        // Perform the operation
        torch::Tensor result;
        
        if (use_inplace && input.defined() && !input.is_sparse()) {
            // In-place operation
            try {
                input.addcmul_(tensor1, tensor2, value);
                result = input;
            } catch (const c10::Error& e) {
                // In-place might fail due to broadcasting or dtype issues
                // Fall back to out-of-place
                result = torch::addcmul(input, tensor1, tensor2, value);
            }
        } else if (use_out_tensor) {
            // Create output tensor for out-of-place operation
            torch::Tensor out;
            
            if (offset < size) {
                uint8_t out_type = data[offset++];
                if (out_type % 3 == 0) {
                    // Use empty tensor with correct shape
                    try {
                        auto result_shape = torch::broadcast_shapes({input.sizes(), tensor1.sizes(), tensor2.sizes()});
                        out = torch::empty(result_shape, input.options());
                    } catch (...) {
                        out = torch::empty_like(input);
                    }
                } else if (out_type % 3 == 1) {
                    // Use existing tensor (input)
                    out = input.clone();
                } else {
                    // Create new tensor from fuzzer data
                    out = createTensorWithOptions(data, offset, size, allow_non_contiguous);
                }
            } else {
                out = torch::empty_like(input);
            }
            
            try {
                torch::addcmul_out(out, input, tensor1, tensor2, value);
                result = out;
            } catch (const c10::Error& e) {
                // Out operation might fail, fall back to regular
                result = torch::addcmul(input, tensor1, tensor2, value);
            }
        } else {
            // Standard operation
            result = torch::addcmul(input, tensor1, tensor2, value);
        }
        
        // Additional operations to increase coverage
        if (result.defined() && offset < size) {
            uint8_t post_op = data[offset++];
            
            switch (post_op % 8) {
                case 0:
                    // Check result properties
                    result.is_contiguous();
                    result.numel();
                    result.element_size();
                    break;
                case 1:
                    // Convert to different dtype if possible
                    if (result.dtype() != torch::kFloat32) {
                        result = result.to(torch::kFloat32);
                    }
                    break;
                case 2:
                    // Compute statistics
                    if (result.numel() > 0 && !result.is_complex()) {
                        torch::mean(result);
                        torch::std(result);
                    }
                    break;
                case 3:
                    // Test with zero-dim tensors
                    if (result.dim() == 0) {
                        result.item<double>();
                    }
                    break;
                case 4:
                    // Test backward if requires_grad
                    if (input.requires_grad() && result.requires_grad()) {
                        try {
                            result.sum().backward();
                        } catch (...) {
                            // Backward might fail for various reasons
                        }
                    }
                    break;
                case 5:
                    // Test with different memory formats
                    if (result.dim() == 4) {
                        result = result.to(torch::MemoryFormat::ChannelsLast);
                    }
                    break;
                case 6:
                    // Test special values
                    if (!result.is_complex()) {
                        torch::isnan(result);
                        torch::isinf(result);
                    }
                    break;
                case 7:
                    // Clone and compare
                    auto cloned = result.clone();
                    torch::equal(result, cloned);
                    break;
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch errors are expected during fuzzing
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}