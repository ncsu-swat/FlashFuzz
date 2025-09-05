#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from the fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) {
        return false;
    }
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) {
        return 0;
    }

    size_t offset = 0;
    
    try {
        // Consume configuration bytes
        uint8_t rank;
        if (!consumeBytes(data, size, offset, rank)) return 0;
        rank = (rank % 5) + 1; // Limit rank to 1-5 dimensions
        
        uint8_t dtype_choice;
        if (!consumeBytes(data, size, offset, dtype_choice)) return 0;
        
        uint8_t requires_grad;
        if (!consumeBytes(data, size, offset, requires_grad)) return 0;
        
        uint8_t use_out_tensor;
        if (!consumeBytes(data, size, offset, use_out_tensor)) return 0;
        
        // Build tensor shape
        std::vector<int64_t> shape;
        int64_t total_elements = 1;
        for (size_t i = 0; i < rank; ++i) {
            uint8_t dim_size;
            if (!consumeBytes(data, size, offset, dim_size)) {
                shape.push_back(1);
            } else {
                // Allow 0-sized dimensions and various sizes
                int64_t dim = (dim_size % 10);
                shape.push_back(dim);
                total_elements *= dim;
            }
        }
        
        // Limit total elements to prevent OOM
        if (total_elements > 10000) {
            return 0;
        }
        
        // Select dtype
        torch::ScalarType scalar_type;
        switch (dtype_choice % 4) {
            case 0:
                scalar_type = torch::kFloat32;
                break;
            case 1:
                scalar_type = torch::kFloat64;
                break;
            case 2:
                scalar_type = torch::kFloat16;
                break;
            case 3:
                scalar_type = torch::kBFloat16;
                break;
            default:
                scalar_type = torch::kFloat32;
        }
        
        // Create input tensor
        torch::Tensor input;
        
        if (total_elements == 0) {
            // Create empty tensor
            input = torch::empty(shape, torch::dtype(scalar_type));
        } else {
            // Fill tensor with fuzzer data
            size_t bytes_per_element = torch::elementSize(scalar_type);
            size_t bytes_needed = total_elements * bytes_per_element;
            
            if (offset + bytes_needed <= size) {
                // Use fuzzer data directly
                input = torch::from_blob(
                    const_cast<uint8_t*>(data + offset),
                    shape,
                    torch::dtype(scalar_type)
                ).clone();
                offset += bytes_needed;
            } else {
                // Generate random data with seed from fuzzer
                uint32_t seed;
                if (consumeBytes(data, size, offset, seed)) {
                    torch::manual_seed(seed);
                }
                input = torch::randn(shape, torch::dtype(scalar_type));
                
                // Scale values to test different ranges
                uint8_t scale_factor;
                if (consumeBytes(data, size, offset, scale_factor)) {
                    float scale = (scale_factor % 100) - 50.0f;
                    input = input * scale;
                }
            }
        }
        
        // Set requires_grad if applicable
        if ((requires_grad & 1) && scalar_type != torch::kFloat16 && 
            scalar_type != torch::kBFloat16) {
            input.requires_grad_(true);
        }
        
        // Test with strided tensors
        uint8_t make_strided;
        if (consumeBytes(data, size, offset, make_strided) && (make_strided & 1) && 
            input.numel() > 1) {
            // Transpose or permute to create non-contiguous tensor
            if (input.dim() >= 2) {
                input = input.transpose(0, 1);
            }
        }
        
        // Call torch.special.expm1
        torch::Tensor result;
        
        if (use_out_tensor & 1) {
            // Test with out parameter
            torch::Tensor out = torch::empty_like(input);
            
            // Test with different out tensor configurations
            uint8_t out_config;
            if (consumeBytes(data, size, offset, out_config)) {
                switch (out_config % 3) {
                    case 0:
                        // Same as input
                        break;
                    case 1:
                        // Different dtype if possible
                        if (scalar_type == torch::kFloat32) {
                            out = torch::empty_like(input, torch::dtype(torch::kFloat64));
                        }
                        break;
                    case 2:
                        // Use input as out (in-place)
                        out = input;
                        break;
                }
            }
            
            result = torch::special::expm1_out(out, input);
        } else {
            result = torch::special::expm1(input);
        }
        
        // Exercise the result tensor
        if (result.numel() > 0) {
            // Check for NaN/Inf
            auto has_nan = result.isnan().any().item<bool>();
            auto has_inf = result.isinf().any().item<bool>();
            
            // Compute some statistics to ensure computation happened
            if (!has_nan && !has_inf && result.numel() > 1) {
                auto sum = result.sum();
                auto mean = result.mean();
                auto max_val = result.max();
                auto min_val = result.min();
            }
        }
        
        // Test backward if gradient is enabled
        if (input.requires_grad() && result.requires_grad()) {
            uint8_t do_backward;
            if (consumeBytes(data, size, offset, do_backward) && (do_backward & 1)) {
                auto grad_output = torch::ones_like(result);
                result.backward(grad_output);
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