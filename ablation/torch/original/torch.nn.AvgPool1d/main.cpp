#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

// Helper to consume values from fuzzer data
template<typename T>
T consume_value(const uint8_t* data, size_t& offset, size_t size, T min_val, T max_val) {
    if (offset + sizeof(T) > size) {
        offset = size;
        return min_val;
    }
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    
    // Ensure value is in valid range
    if (value < min_val) value = min_val;
    if (value > max_val) value = max_val;
    return value;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) {  // Need minimum bytes for configuration
        return 0;
    }

    try {
        size_t offset = 0;
        
        // Parse AvgPool1d parameters
        int64_t kernel_size = consume_value<int64_t>(data, offset, size, 1, 100);
        int64_t stride = consume_value<int64_t>(data, offset, size, 0, 100);
        int64_t padding = consume_value<int64_t>(data, offset, size, 0, 50);
        
        // If stride is 0, use kernel_size as default (as per PyTorch behavior)
        if (stride == 0) {
            stride = kernel_size;
        }
        
        // Parse boolean flags
        bool ceil_mode = (offset < size) ? (data[offset++] & 1) : false;
        bool count_include_pad = (offset < size) ? (data[offset++] & 1) : true;
        
        // Determine if we should test with 2D (C,L) or 3D (N,C,L) input
        bool use_batch = (offset < size) ? (data[offset++] & 1) : true;
        
#ifdef DEBUG_FUZZ
        std::cout << "AvgPool1d params: kernel_size=" << kernel_size 
                  << ", stride=" << stride << ", padding=" << padding
                  << ", ceil_mode=" << ceil_mode 
                  << ", count_include_pad=" << count_include_pad
                  << ", use_batch=" << use_batch << std::endl;
#endif
        
        // Create AvgPool1d module with options
        torch::nn::AvgPool1dOptions options(kernel_size);
        options.stride(stride)
               .padding(padding)
               .ceil_mode(ceil_mode)
               .count_include_pad(count_include_pad);
        
        torch::nn::AvgPool1d pool(options);
        
        // Create input tensor
        torch::Tensor input;
        
        // Try to parse a tensor from remaining data
        if (offset < size) {
            try {
                input = fuzzer_utils::createTensor(data, size, offset);
                
                // Reshape tensor to appropriate dimensions for AvgPool1d
                int64_t total_elements = input.numel();
                
                if (total_elements > 0) {
                    if (use_batch) {
                        // Reshape to (N, C, L) format
                        // Use modulo to ensure valid dimensions
                        int64_t N = 1 + (total_elements % 4);
                        int64_t C = 1 + ((total_elements / N) % 8);
                        int64_t L = total_elements / (N * C);
                        
                        if (L > 0) {
                            input = input.reshape({N, C, L});
                        } else {
                            // Fallback to minimal valid shape
                            input = torch::randn({1, 1, kernel_size + 1});
                        }
                    } else {
                        // Reshape to (C, L) format
                        int64_t C = 1 + (total_elements % 8);
                        int64_t L = total_elements / C;
                        
                        if (L > 0) {
                            input = input.reshape({C, L});
                        } else {
                            // Fallback to minimal valid shape
                            input = torch::randn({1, kernel_size + 1});
                        }
                    }
                }
            } catch (const std::exception& e) {
                // If tensor creation fails, create a random tensor
#ifdef DEBUG_FUZZ
                std::cout << "Tensor creation failed: " << e.what() << ", using random tensor" << std::endl;
#endif
                if (use_batch) {
                    input = torch::randn({2, 3, kernel_size * 2 + padding * 2 + 1});
                } else {
                    input = torch::randn({3, kernel_size * 2 + padding * 2 + 1});
                }
            }
        } else {
            // Create random tensor if no data left
            if (use_batch) {
                input = torch::randn({1, 2, kernel_size + 2 * padding + 1});
            } else {
                input = torch::randn({2, kernel_size + 2 * padding + 1});
            }
        }
        
#ifdef DEBUG_FUZZ
        std::cout << "Input shape: " << input.sizes() 
                  << ", dtype: " << input.dtype() << std::endl;
#endif
        
        // Test the forward pass
        torch::Tensor output = pool->forward(input);
        
#ifdef DEBUG_FUZZ
        std::cout << "Output shape: " << output.sizes() << std::endl;
#endif
        
        // Verify output shape calculation
        int64_t L_in = use_batch ? input.size(2) : input.size(1);
        int64_t L_out_expected;
        
        if (ceil_mode) {
            L_out_expected = (L_in + 2 * padding - kernel_size + stride - 1) / stride + 1;
        } else {
            L_out_expected = (L_in + 2 * padding - kernel_size) / stride + 1;
        }
        
        // Ensure L_out is at least 0
        L_out_expected = std::max(L_out_expected, int64_t(0));
        
        int64_t L_out_actual = use_batch ? output.size(2) : output.size(1);
        
        if (L_out_actual != L_out_expected && L_out_expected >= 0) {
#ifdef DEBUG_FUZZ
            std::cout << "Warning: Output length mismatch. Expected: " << L_out_expected 
                      << ", Actual: " << L_out_actual << std::endl;
#endif
        }
        
        // Additional edge case testing
        
        // Test with zero-sized dimensions
        if (offset < size && (data[offset] & 1)) {
            try {
                torch::Tensor zero_input = use_batch ? 
                    torch::zeros({0, 1, kernel_size}) : 
                    torch::zeros({1, 0});
                torch::Tensor zero_output = pool->forward(zero_input);
#ifdef DEBUG_FUZZ
                std::cout << "Zero-dim test passed. Output shape: " << zero_output.sizes() << std::endl;
#endif
            } catch (const std::exception& e) {
                // Expected for some invalid configurations
#ifdef DEBUG_FUZZ
                std::cout << "Zero-dim test exception (expected): " << e.what() << std::endl;
#endif
            }
        }
        
        // Test with very small input relative to kernel
        if (offset < size && (data[offset] & 2)) {
            try {
                int64_t small_L = std::max(int64_t(1), kernel_size - padding);
                torch::Tensor small_input = use_batch ? 
                    torch::randn({1, 1, small_L}) : 
                    torch::randn({1, small_L});
                torch::Tensor small_output = pool->forward(small_input);
#ifdef DEBUG_FUZZ
                std::cout << "Small input test passed. Input L: " << small_L 
                          << ", Output shape: " << small_output.sizes() << std::endl;
#endif
            } catch (const std::exception& e) {
#ifdef DEBUG_FUZZ
                std::cout << "Small input test exception: " << e.what() << std::endl;
#endif
            }
        }
        
        // Test gradient computation if input requires grad
        if (input.requires_grad()) {
            try {
                output.backward(torch::ones_like(output));
#ifdef DEBUG_FUZZ
                std::cout << "Backward pass successful" << std::endl;
#endif
            } catch (const std::exception& e) {
#ifdef DEBUG_FUZZ
                std::cout << "Backward pass exception: " << e.what() << std::endl;
#endif
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors
#ifdef DEBUG_FUZZ
        std::cout << "PyTorch error: " << e.what() << std::endl;
#endif
        return 0;  // Continue fuzzing
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;  // Discard this input
    } catch (...) {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}