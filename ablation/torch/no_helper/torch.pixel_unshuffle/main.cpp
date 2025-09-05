#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 16) {
        return 0;  // Need minimum bytes for basic parameters
    }

    try {
        // Parse fuzzer input
        size_t offset = 0;
        
        // Extract downscale_factor (1-10 range to avoid extreme memory usage)
        int downscale_factor = 1 + (data[offset++] % 10);
        
        // Extract number of dimensions (3-5 for pixel_unshuffle: batch, channels, height, width)
        int ndims = 3 + (data[offset++] % 3);
        
        // Extract dtype selector
        int dtype_selector = data[offset++] % 6;
        
        // Extract device selector
        bool use_cuda = (data[offset++] % 2) == 1 && torch::cuda::is_available();
        
        // Extract requires_grad
        bool requires_grad = (data[offset++] % 2) == 1;
        
        // Build tensor shape
        std::vector<int64_t> shape;
        for (int i = 0; i < ndims; i++) {
            if (offset >= size) break;
            
            // For height and width dimensions, ensure they're divisible by downscale_factor
            int64_t dim_size;
            if (i >= ndims - 2) {
                // Last two dimensions (H, W) must be divisible by downscale_factor
                dim_size = downscale_factor * (1 + (data[offset++] % 16));
            } else if (i == ndims - 3) {
                // Channels dimension
                dim_size = 1 + (data[offset++] % 32);
            } else {
                // Batch dimensions
                dim_size = 1 + (data[offset++] % 8);
            }
            shape.push_back(dim_size);
        }
        
        // Ensure we have at least 3 dimensions
        while (shape.size() < 3) {
            shape.push_back(downscale_factor * 2);
        }
        
        // Calculate total elements
        int64_t total_elements = 1;
        for (auto s : shape) {
            total_elements *= s;
        }
        
        // Limit total elements to prevent OOM
        if (total_elements > 1000000) {
            return 0;
        }
        
        // Create tensor options
        torch::TensorOptions options;
        switch (dtype_selector) {
            case 0: options = options.dtype(torch::kFloat32); break;
            case 1: options = options.dtype(torch::kFloat64); break;
            case 2: options = options.dtype(torch::kFloat16); break;
            case 3: options = options.dtype(torch::kBFloat16); break;
            case 4: options = options.dtype(torch::kInt32); break;
            case 5: options = options.dtype(torch::kInt64); break;
            default: options = options.dtype(torch::kFloat32); break;
        }
        
        if (use_cuda) {
            options = options.device(torch::kCUDA);
        } else {
            options = options.device(torch::kCPU);
        }
        
        options = options.requires_grad(requires_grad && (dtype_selector <= 3));
        
        // Create input tensor
        torch::Tensor input;
        
        // Decide how to initialize the tensor
        if (offset < size && data[offset++] % 3 == 0) {
            // Random initialization
            input = torch::randn(shape, options);
        } else if (offset < size && data[offset++] % 2 == 0) {
            // Zeros
            input = torch::zeros(shape, options);
        } else {
            // Ones
            input = torch::ones(shape, options);
        }
        
        // Add some variations to the tensor
        if (offset < size) {
            uint8_t variation = data[offset++];
            if (variation % 4 == 0 && input.numel() > 0) {
                // Make contiguous
                input = input.contiguous();
            } else if (variation % 4 == 1 && shape.size() >= 2) {
                // Transpose last two dims
                input = input.transpose(-1, -2).contiguous();
            } else if (variation % 4 == 2 && input.numel() > 0) {
                // Add noise
                input = input + torch::randn_like(input) * 0.1;
            }
        }
        
        // Apply pixel_unshuffle
        torch::Tensor output = torch::nn::functional::pixel_unshuffle(input, downscale_factor);
        
        // Perform some operations on output to trigger more code paths
        if (offset < size && data[offset++] % 4 == 0) {
            auto sum = output.sum();
        }
        if (offset < size && data[offset++] % 4 == 0) {
            auto mean = output.mean();
        }
        if (offset < size && data[offset++] % 4 == 0 && output.requires_grad()) {
            auto grad_output = torch::ones_like(output);
            output.backward(grad_output);
        }
        
        // Try edge cases
        if (offset < size && data[offset++] % 10 == 0) {
            // Try with empty tensor
            torch::Tensor empty_tensor = torch::empty({0, 1, downscale_factor, downscale_factor}, options);
            try {
                auto empty_result = torch::nn::functional::pixel_unshuffle(empty_tensor, downscale_factor);
            } catch (...) {
                // Ignore errors on edge cases
            }
        }
        
        if (offset < size && data[offset++] % 10 == 0) {
            // Try with single element tensors
            torch::Tensor single = torch::ones({1, 1, downscale_factor, downscale_factor}, options);
            try {
                auto single_result = torch::nn::functional::pixel_unshuffle(single, downscale_factor);
            } catch (...) {
                // Ignore errors on edge cases
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch errors are expected for invalid inputs
        return 0;
    } catch (const std::bad_alloc& e) {
        // Memory allocation failures
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Unknown errors
        return -1;
    }
    
    return 0;
}