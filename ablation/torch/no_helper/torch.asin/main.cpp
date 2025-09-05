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
        
        // Extract basic parameters from fuzzer input
        uint8_t rank = (size > offset) ? data[offset++] % 5 : 0; // Limit rank to 0-4
        uint8_t dtype_idx = (size > offset) ? data[offset++] % 6 : 0;
        bool requires_grad = (size > offset) ? (data[offset++] & 1) : false;
        bool use_out_tensor = (size > offset) ? (data[offset++] & 1) : false;
        
        // Determine dtype
        torch::ScalarType dtype;
        switch (dtype_idx) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            case 3: dtype = torch::kBFloat16; break;
            case 4: dtype = torch::kComplex64; break;
            case 5: dtype = torch::kComplex128; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Build shape vector
        std::vector<int64_t> shape;
        for (uint8_t i = 0; i < rank && offset < size; ++i) {
            int64_t dim = data[offset++] % 10; // Keep dimensions small
            shape.push_back(dim);
        }
        
        // Calculate total elements
        int64_t num_elements = 1;
        for (auto dim : shape) {
            num_elements *= dim;
        }
        
        // Limit total elements to prevent OOM
        if (num_elements > 10000) {
            num_elements = 10000;
            if (!shape.empty()) {
                shape[0] = std::min(shape[0], (int64_t)10000);
            }
        }
        
        // Create input tensor
        torch::Tensor input;
        
        if (num_elements == 0 || shape.empty()) {
            // Handle empty tensor case
            input = torch::empty({0}, torch::TensorOptions().dtype(dtype).requires_grad(requires_grad));
        } else {
            // Fill tensor with values from fuzzer data
            std::vector<float> values;
            values.reserve(num_elements);
            
            for (int64_t i = 0; i < num_elements && offset < size; ++i) {
                // Map byte to range [-1.5, 1.5] to include valid and invalid asin inputs
                float val = (data[offset++] / 127.5f) - 1.0f;
                val *= 1.5f; // Extend range to include values outside [-1, 1]
                values.push_back(val);
            }
            
            // Fill remaining with random valid/invalid values
            while (values.size() < num_elements) {
                float val = ((offset < size ? data[offset++] : rand() % 256) / 127.5f) - 1.0f;
                val *= 1.5f;
                values.push_back(val);
            }
            
            // Create tensor from values
            auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(requires_grad);
            input = torch::from_blob(values.data(), {num_elements}, options).clone();
            
            if (!shape.empty()) {
                input = input.reshape(shape);
            }
            
            // Convert to target dtype
            if (dtype != torch::kFloat32) {
                input = input.to(dtype);
            }
        }
        
        // Test different tensor layouts
        if (offset < size && (data[offset++] & 1) && input.numel() > 0) {
            // Make tensor non-contiguous
            if (input.dim() >= 2) {
                input = input.transpose(0, input.dim() - 1);
            }
        }
        
        // Apply torch.asin
        torch::Tensor result;
        
        if (use_out_tensor && offset < size) {
            // Test with out parameter
            torch::Tensor out;
            
            // Decide out tensor configuration
            uint8_t out_config = data[offset++] % 4;
            switch (out_config) {
                case 0:
                    // Same shape and dtype
                    out = torch::empty_like(input);
                    break;
                case 1:
                    // Different dtype
                    out = torch::empty_like(input, torch::TensorOptions().dtype(torch::kFloat64));
                    break;
                case 2:
                    // Wrong shape (should trigger error)
                    if (input.numel() > 1) {
                        out = torch::empty({input.numel() + 1}, input.options());
                    } else {
                        out = torch::empty_like(input);
                    }
                    break;
                default:
                    out = torch::empty_like(input);
                    break;
            }
            
            result = torch::asin_out(out, input);
        } else {
            result = torch::asin(input);
        }
        
        // Exercise the result tensor
        if (result.numel() > 0) {
            // Check for NaN values (expected for inputs outside [-1, 1])
            auto has_nan = torch::isnan(result).any().item<bool>();
            
            // Try backward pass if gradients enabled
            if (requires_grad && input.requires_grad() && result.requires_grad()) {
                try {
                    auto grad_output = torch::ones_like(result);
                    result.backward(grad_output);
                } catch (...) {
                    // Backward might fail for NaN values, which is expected
                }
            }
            
            // Access some elements to ensure computation completed
            if (result.numel() > 0) {
                result.flatten()[0];
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid operations
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