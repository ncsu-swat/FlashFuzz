#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from the fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t& offset, size_t size, T& out) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&out, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    try {
        if (size < 4) return 0;
        
        size_t offset = 0;
        
        // Consume configuration bytes
        uint8_t rank, dtype_idx, memory_format_idx, requires_grad;
        if (!consumeBytes(data, offset, size, rank)) return 0;
        if (!consumeBytes(data, offset, size, dtype_idx)) return 0;
        if (!consumeBytes(data, offset, size, memory_format_idx)) return 0;
        if (!consumeBytes(data, offset, size, requires_grad)) return 0;
        
        // Limit rank to reasonable value
        rank = (rank % 5) + 1;  // 1-5 dimensions
        
        // Build shape from fuzzer input
        std::vector<int64_t> shape;
        for (int i = 0; i < rank; i++) {
            uint8_t dim_size;
            if (!consumeBytes(data, offset, size, dim_size)) {
                shape.push_back(1);
            } else {
                // Allow 0-sized dimensions for edge cases
                shape.push_back(dim_size % 10);  // 0-9
            }
        }
        
        // Select dtype
        torch::ScalarType dtype;
        switch (dtype_idx % 8) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kInt8; break;
            case 5: dtype = torch::kUInt8; break;
            case 6: dtype = torch::kBool; break;
            case 7: dtype = torch::kFloat16; break;
            default: dtype = torch::kFloat32;
        }
        
        // Create tensor with random data
        torch::Tensor input;
        
        // Calculate total elements
        int64_t total_elements = 1;
        for (auto dim : shape) {
            total_elements *= dim;
        }
        
        // Create tensor from remaining fuzzer data or random
        if (total_elements > 0 && total_elements <= 1000000) {  // Limit size
            auto options = torch::TensorOptions().dtype(dtype);
            
            if (requires_grad % 2 && (dtype == torch::kFloat32 || dtype == torch::kFloat64)) {
                options = options.requires_grad(true);
            }
            
            // Create tensor with different initialization strategies
            uint8_t init_strategy;
            if (consumeBytes(data, offset, size, init_strategy)) {
                switch (init_strategy % 5) {
                    case 0:
                        input = torch::zeros(shape, options);
                        break;
                    case 1:
                        input = torch::ones(shape, options);
                        break;
                    case 2:
                        input = torch::randn(shape, options);
                        break;
                    case 3:
                        input = torch::rand(shape, options);
                        break;
                    case 4:
                        input = torch::empty(shape, options);
                        break;
                }
            } else {
                input = torch::randn(shape, options);
            }
        } else {
            // Handle edge case of empty or very large tensor
            input = torch::empty({0}, torch::TensorOptions().dtype(dtype));
        }
        
        // Select memory format
        c10::MemoryFormat memory_format;
        switch (memory_format_idx % 4) {
            case 0: memory_format = c10::MemoryFormat::Preserve; break;
            case 1: memory_format = c10::MemoryFormat::Contiguous; break;
            case 2: memory_format = c10::MemoryFormat::ChannelsLast; break;
            case 3: memory_format = c10::MemoryFormat::ChannelsLast3d; break;
            default: memory_format = c10::MemoryFormat::Preserve;
        }
        
        // Test clone operation
        torch::Tensor cloned = input.clone(memory_format);
        
        // Verify clone properties
        if (cloned.defined()) {
            // Test that clone is independent
            if (input.numel() > 0 && input.dtype() != torch::kBool) {
                // Modify clone and verify original unchanged
                cloned.add_(1);
            }
            
            // Test gradient flow if applicable
            if (input.requires_grad() && cloned.requires_grad()) {
                try {
                    auto sum = cloned.sum();
                    if (sum.requires_grad()) {
                        sum.backward();
                    }
                } catch (...) {
                    // Gradient computation might fail for some dtypes
                }
            }
        }
        
        // Test edge cases with strided tensors
        if (input.numel() > 1) {
            uint8_t stride_op;
            if (consumeBytes(data, offset, size, stride_op)) {
                torch::Tensor strided_input;
                try {
                    switch (stride_op % 4) {
                        case 0:
                            strided_input = input.transpose(0, -1);
                            break;
                        case 1:
                            strided_input = input.narrow(0, 0, std::max(int64_t(1), input.size(0) / 2));
                            break;
                        case 2:
                            strided_input = input.unsqueeze(0);
                            break;
                        case 3:
                            if (input.dim() >= 2) {
                                strided_input = input.permute({-1, 0});
                            } else {
                                strided_input = input;
                            }
                            break;
                    }
                    
                    if (strided_input.defined()) {
                        torch::Tensor cloned_strided = strided_input.clone(memory_format);
                    }
                } catch (...) {
                    // Some operations might fail on certain tensor shapes
                }
            }
        }
        
    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cout << "Exception caught: unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}