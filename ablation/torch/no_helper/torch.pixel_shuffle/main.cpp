#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 16) {
        return 0;  // Need minimum bytes for configuration
    }

    try {
        // Parse configuration from fuzzer input
        size_t offset = 0;
        
        // Extract upscale_factor (1-16 range to avoid extreme memory usage)
        int upscale_factor = 1 + (data[offset++] % 16);
        
        // Extract number of dimensions (3-5 typical for pixel_shuffle)
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
            
            int64_t dim_size;
            if (i == ndims - 3) {
                // Channel dimension: must be divisible by upscale_factor^2
                int base_channels = 1 + (data[offset++] % 8);
                dim_size = base_channels * upscale_factor * upscale_factor;
            } else if (i < ndims - 3) {
                // Batch dimensions
                dim_size = 1 + (data[offset++] % 4);
            } else {
                // Spatial dimensions (H, W)
                dim_size = 1 + (data[offset++] % 32);
            }
            shape.push_back(dim_size);
        }
        
        // Ensure we have at least 3 dimensions
        while (shape.size() < 3) {
            shape.push_back(1);
        }
        
        // Select dtype
        torch::ScalarType scalar_type;
        switch (dtype_selector) {
            case 0: scalar_type = torch::kFloat32; break;
            case 1: scalar_type = torch::kFloat64; break;
            case 2: scalar_type = torch::kFloat16; break;
            case 3: scalar_type = torch::kBFloat16; break;
            case 4: scalar_type = torch::kInt32; break;
            case 5: scalar_type = torch::kInt64; break;
            default: scalar_type = torch::kFloat32; break;
        }
        
        // Create options
        auto options = torch::TensorOptions().dtype(scalar_type);
        if (use_cuda) {
            options = options.device(torch::kCUDA);
        }
        if (requires_grad && (scalar_type == torch::kFloat32 || 
                              scalar_type == torch::kFloat64 ||
                              scalar_type == torch::kFloat16 ||
                              scalar_type == torch::kBFloat16)) {
            options = options.requires_grad(true);
        }
        
        // Create input tensor
        torch::Tensor input;
        
        // Decide initialization method
        if (offset < size) {
            int init_method = data[offset++] % 5;
            switch (init_method) {
                case 0:
                    input = torch::randn(shape, options);
                    break;
                case 1:
                    input = torch::ones(shape, options);
                    break;
                case 2:
                    input = torch::zeros(shape, options);
                    break;
                case 3:
                    input = torch::empty(shape, options);
                    if (scalar_type == torch::kFloat32 || scalar_type == torch::kFloat64) {
                        input.uniform_(-1.0, 1.0);
                    }
                    break;
                case 4:
                    // Create from remaining data
                    {
                        size_t numel = 1;
                        for (auto d : shape) numel *= d;
                        
                        if (scalar_type == torch::kFloat32 || scalar_type == torch::kFloat64) {
                            std::vector<float> values;
                            while (offset < size && values.size() < numel) {
                                float val = static_cast<float>(data[offset++]) / 255.0f;
                                values.push_back(val);
                            }
                            while (values.size() < numel) {
                                values.push_back(0.0f);
                            }
                            input = torch::from_blob(values.data(), shape, torch::kFloat32).clone().to(options);
                        } else {
                            input = torch::randint(0, 256, shape, options);
                        }
                    }
                    break;
                default:
                    input = torch::randn(shape, options);
            }
        } else {
            input = torch::randn(shape, options);
        }
        
        // Test with contiguous tensor
        torch::Tensor output = torch::pixel_shuffle(input, upscale_factor);
        
        // Test with non-contiguous tensor if we have extra bytes
        if (offset < size && (data[offset++] % 2) == 1) {
            torch::Tensor transposed = input.transpose(-1, -2);
            torch::Tensor output2 = torch::pixel_shuffle(transposed.contiguous(), upscale_factor);
        }
        
        // Test with view if possible
        if (offset < size && (data[offset++] % 2) == 1 && shape.size() > 3) {
            auto view_shape = shape;
            if (view_shape[0] > 1) {
                view_shape[0] = 1;
                view_shape.insert(view_shape.begin(), shape[0]);
                torch::Tensor viewed = input.view(view_shape);
                torch::Tensor output3 = torch::pixel_shuffle(viewed, upscale_factor);
            }
        }
        
        // Test backward pass if applicable
        if (requires_grad && output.requires_grad()) {
            torch::Tensor grad_output = torch::ones_like(output);
            output.backward(grad_output);
        }
        
        // Test edge cases with different upscale factors
        if (offset < size) {
            int alt_factor = 1 + (data[offset++] % 8);
            // Adjust channel dimension for new factor
            auto alt_shape = shape;
            if (alt_shape.size() >= 3) {
                alt_shape[alt_shape.size() - 3] = alt_factor * alt_factor * 
                    (1 + (alt_shape[alt_shape.size() - 3] / (upscale_factor * upscale_factor)));
            }
            torch::Tensor alt_input = torch::randn(alt_shape, options);
            torch::Tensor alt_output = torch::pixel_shuffle(alt_input, alt_factor);
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid configurations
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Unknown exception
        return -1;
    }
    
    return 0;
}