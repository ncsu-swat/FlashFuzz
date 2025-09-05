#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) {
        return 0;
    }

    try {
        size_t offset = 0;

        // Extract configuration bytes
        uint8_t rank = data[offset++] % 5;  // 0-4 dimensions
        uint8_t dtype_selector = data[offset++] % 6;  // Select from common dtypes
        uint8_t requires_grad = data[offset++] & 1;
        uint8_t use_complex = data[offset++] & 1;
        
        // Determine dtype
        torch::ScalarType dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            case 3: dtype = torch::kBFloat16; break;
            case 4: dtype = torch::kInt32; break;
            case 5: dtype = torch::kInt64; break;
            default: dtype = torch::kFloat32; break;
        }

        // For complex types, override if requested and dtype is floating
        if (use_complex && (dtype == torch::kFloat32 || dtype == torch::kFloat64)) {
            dtype = (dtype == torch::kFloat32) ? torch::kComplexFloat : torch::kComplexDouble;
        }

        // Build shape vector
        std::vector<int64_t> shape;
        size_t total_elements = 1;
        for (uint8_t i = 0; i < rank; ++i) {
            if (offset >= size) break;
            int64_t dim = (data[offset++] % 8) + 1;  // 1-8 per dimension
            shape.push_back(dim);
            total_elements *= dim;
        }

        // Limit total elements to prevent OOM
        if (total_elements > 10000) {
            total_elements = 10000;
            if (!shape.empty()) {
                shape[0] = 10000 / (total_elements / shape[0]);
                if (shape[0] == 0) shape[0] = 1;
            }
        }

        // Create tensor with various initialization methods
        torch::Tensor input;
        if (offset < size) {
            uint8_t init_method = data[offset++] % 5;
            switch (init_method) {
                case 0:
                    input = torch::zeros(shape, torch::dtype(dtype));
                    break;
                case 1:
                    input = torch::ones(shape, torch::dtype(dtype));
                    break;
                case 2:
                    input = torch::randn(shape, torch::dtype(dtype));
                    break;
                case 3:
                    input = torch::rand(shape, torch::dtype(dtype));
                    break;
                case 4:
                    // Create from remaining data
                    if (shape.empty()) {
                        input = torch::scalar_tensor(0.5, torch::dtype(dtype));
                    } else {
                        input = torch::empty(shape, torch::dtype(dtype));
                        // Fill with fuzz data if possible
                        if (offset < size && input.is_floating_point()) {
                            auto input_flat = input.flatten();
                            for (int64_t i = 0; i < input_flat.numel() && offset < size; ++i) {
                                float val = static_cast<float>(data[offset++]) / 255.0f;
                                val = (val - 0.5f) * 100.0f;  // Scale to [-50, 50]
                                input_flat[i] = val;
                            }
                            input = input_flat.reshape(shape);
                        }
                    }
                    break;
                default:
                    input = torch::zeros(shape, torch::dtype(dtype));
            }
        } else {
            input = torch::zeros(shape, torch::dtype(dtype));
        }

        // Set requires_grad if applicable
        if (requires_grad && input.is_floating_point() && !input.is_complex()) {
            input.requires_grad_(true);
        }

        // Test edge cases
        if (offset < size) {
            uint8_t edge_case = data[offset++] % 4;
            switch (edge_case) {
                case 0:
                    // Add NaN values if floating point
                    if (input.is_floating_point() && input.numel() > 0) {
                        input.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
                    }
                    break;
                case 1:
                    // Add infinity values if floating point
                    if (input.is_floating_point() && input.numel() > 0) {
                        input.flatten()[0] = std::numeric_limits<float>::infinity();
                        if (input.numel() > 1) {
                            input.flatten()[1] = -std::numeric_limits<float>::infinity();
                        }
                    }
                    break;
                case 2:
                    // Very large values
                    if (input.is_floating_point() && input.numel() > 0) {
                        input = input * 1e10;
                    }
                    break;
                case 3:
                    // Very small values
                    if (input.is_floating_point() && input.numel() > 0) {
                        input = input * 1e-10;
                    }
                    break;
            }
        }

        // Call torch.special.round
        torch::Tensor result = torch::special::round(input);

        // Additional operations to exercise more paths
        if (offset < size && data[offset++] & 1) {
            // In-place operation if possible
            if (input.is_floating_point()) {
                torch::Tensor input_copy = input.clone();
                input_copy = torch::special::round(input_copy);
            }
        }

        // Test with output tensor
        if (offset < size && data[offset++] & 1) {
            torch::Tensor out = torch::empty_like(input);
            torch::special::round(input, out);
        }

        // Verify result properties
        if (result.defined()) {
            auto result_size = result.sizes();
            auto result_dtype = result.dtype();
            auto result_device = result.device();
            
            // Access some elements if tensor is not empty
            if (result.numel() > 0) {
                if (result.is_floating_point()) {
                    result.item<float>();
                }
            }
        }

    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid operations
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}