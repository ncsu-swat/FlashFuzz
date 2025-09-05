#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <vector>
#include <algorithm>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 16) {
        return 0;  // Need minimum bytes for parameters
    }

    try {
        // Create a stream from the fuzzer data
        size_t offset = 0;
        
        // Helper lambda to consume bytes
        auto consumeBytes = [&](size_t n) -> std::vector<uint8_t> {
            if (offset + n > size) {
                return std::vector<uint8_t>(n, 0);
            }
            std::vector<uint8_t> result(data + offset, data + offset + n);
            offset += n;
            return result;
        };
        
        auto consumeUInt8 = [&]() -> uint8_t {
            if (offset >= size) return 0;
            return data[offset++];
        };
        
        auto consumeUInt32 = [&]() -> uint32_t {
            auto bytes = consumeBytes(4);
            return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) | 
                   (uint32_t(bytes[2]) << 8) | uint32_t(bytes[3]);
        };

        // Parse AvgPool1d parameters
        int kernel_size = 1 + (consumeUInt8() % 10);  // 1-10
        int stride = consumeUInt8() % 10;  // 0-9, 0 means use kernel_size
        if (stride == 0) stride = kernel_size;
        int padding = consumeUInt8() % 5;  // 0-4
        bool ceil_mode = consumeUInt8() & 1;
        bool count_include_pad = consumeUInt8() & 1;
        
        // Parse tensor dimensions
        uint8_t dim_choice = consumeUInt8() % 3;
        std::vector<int64_t> shape;
        
        if (dim_choice == 0) {
            // 2D tensor (C, L)
            int64_t C = 1 + (consumeUInt8() % 16);  // 1-16 channels
            int64_t L = 1 + (consumeUInt8() % 100); // 1-100 length
            shape = {C, L};
        } else if (dim_choice == 1) {
            // 3D tensor (N, C, L)
            int64_t N = 1 + (consumeUInt8() % 8);   // 1-8 batch
            int64_t C = 1 + (consumeUInt8() % 16);  // 1-16 channels
            int64_t L = 1 + (consumeUInt8() % 100); // 1-100 length
            shape = {N, C, L};
        } else {
            // Edge case: very small or zero dimensions
            int64_t N = consumeUInt8() % 3;  // 0-2
            int64_t C = consumeUInt8() % 3;  // 0-2
            int64_t L = consumeUInt8() % 20; // 0-19
            if (N > 0 && C > 0 && L > 0) {
                shape = {N, C, L};
            } else if (C > 0 && L > 0) {
                shape = {C, L};
            } else {
                // Invalid shape, create minimal valid tensor
                shape = {1, 1};
            }
        }
        
        // Choose dtype
        uint8_t dtype_choice = consumeUInt8() % 3;
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Create input tensor
        torch::Tensor input;
        uint8_t tensor_init = consumeUInt8() % 4;
        
        try {
            switch (tensor_init) {
                case 0:
                    input = torch::randn(shape, torch::dtype(dtype));
                    break;
                case 1:
                    input = torch::ones(shape, torch::dtype(dtype));
                    break;
                case 2:
                    input = torch::zeros(shape, torch::dtype(dtype));
                    break;
                case 3: {
                    // Initialize with fuzzer data
                    input = torch::empty(shape, torch::dtype(dtype));
                    auto numel = input.numel();
                    if (numel > 0 && numel < 10000) {  // Limit for performance
                        auto flat = input.flatten();
                        for (int64_t i = 0; i < numel; ++i) {
                            float val = float(consumeUInt8()) / 255.0f * 10.0f - 5.0f;
                            flat[i] = val;
                        }
                        input = flat.reshape(shape);
                    }
                    break;
                }
                default:
                    input = torch::randn(shape, torch::dtype(dtype));
            }
        } catch (...) {
            // If tensor creation fails, use a minimal valid tensor
            input = torch::randn({1, 3}, torch::dtype(torch::kFloat32));
        }
        
        // Test different device configurations
        uint8_t device_choice = consumeUInt8() % 2;
        if (device_choice == 1 && torch::cuda::is_available()) {
            try {
                input = input.to(torch::kCUDA);
            } catch (...) {
                // Keep on CPU if CUDA transfer fails
            }
        }
        
        // Test gradient tracking
        if (consumeUInt8() & 1) {
            if (input.dtype() == torch::kFloat32 || input.dtype() == torch::kFloat64) {
                input.requires_grad_(true);
            }
        }
        
        // Create and apply AvgPool1d
        try {
            torch::nn::AvgPool1d pool(torch::nn::AvgPool1dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
                .ceil_mode(ceil_mode)
                .count_include_pad(count_include_pad));
            
            torch::Tensor output = pool(input);
            
            // Perform additional operations to increase coverage
            if (output.numel() > 0) {
                // Test backward pass if gradients are enabled
                if (input.requires_grad()) {
                    try {
                        auto loss = output.sum();
                        loss.backward();
                    } catch (...) {
                        // Ignore backward errors
                    }
                }
                
                // Access output properties
                auto output_shape = output.sizes();
                auto output_dtype = output.dtype();
                auto is_contiguous = output.is_contiguous();
                
                // Try some tensor operations
                if (consumeUInt8() & 1) {
                    try {
                        auto mean_val = output.mean();
                        auto max_val = output.max();
                        auto min_val = output.min();
                    } catch (...) {
                        // Ignore operation errors
                    }
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid configurations
        } catch (const std::exception& e) {
            // Other exceptions might indicate bugs
            std::cout << "Exception caught: " << e.what() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions
        return -1;
    }
    
    return 0;
}