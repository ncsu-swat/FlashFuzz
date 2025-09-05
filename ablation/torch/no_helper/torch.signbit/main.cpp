#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 4) {
        return 0;  // Need at least some bytes to work with
    }

    try
    {
        size_t offset = 0;
        
        // Extract configuration from fuzzer input
        uint8_t dtype_selector = (offset < size) ? data[offset++] : 0;
        uint8_t ndim = (offset < size) ? (data[offset++] % 5) : 1;  // 0-4 dimensions
        uint8_t use_out_tensor = (offset < size) ? (data[offset++] % 2) : 0;
        uint8_t device_type = (offset < size) ? (data[offset++] % 2) : 0;  // CPU or CUDA
        
        // Build shape vector
        std::vector<int64_t> shape;
        int64_t total_elements = 1;
        for (size_t i = 0; i < ndim; ++i) {
            if (offset >= size) break;
            int64_t dim_size = (data[offset++] % 10);  // Keep dimensions small
            shape.push_back(dim_size);
            total_elements *= dim_size;
        }
        
        // Limit total elements to prevent OOM
        if (total_elements > 10000) {
            total_elements = 10000;
            if (!shape.empty()) {
                shape[0] = std::min(shape[0], (int64_t)10000);
            }
        }
        
        // Select dtype based on fuzzer input
        torch::ScalarType dtype;
        switch (dtype_selector % 8) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            case 3: dtype = torch::kBFloat16; break;
            case 4: dtype = torch::kInt32; break;
            case 5: dtype = torch::kInt64; break;
            case 6: dtype = torch::kInt8; break;
            case 7: dtype = torch::kUInt8; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Create device
        torch::Device device(torch::kCPU);
        if (device_type == 1 && torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
        }
        
        // Create input tensor with various patterns
        torch::Tensor input;
        if (shape.empty()) {
            // Scalar tensor
            if (offset < size) {
                float val = static_cast<float>(data[offset++]) - 128.0f;
                input = torch::tensor(val, torch::TensorOptions().dtype(dtype).device(device));
            } else {
                input = torch::tensor(0.0f, torch::TensorOptions().dtype(dtype).device(device));
            }
        } else {
            // Multi-dimensional tensor
            uint8_t pattern = (offset < size) ? data[offset++] : 0;
            
            switch (pattern % 6) {
                case 0:
                    // Random values from fuzzer data
                    {
                        std::vector<float> values;
                        for (int64_t i = 0; i < total_elements && offset < size; ++i) {
                            float val = static_cast<float>(data[offset++]) - 128.0f;
                            values.push_back(val);
                        }
                        while (values.size() < total_elements) {
                            values.push_back(0.0f);
                        }
                        input = torch::from_blob(values.data(), shape, torch::kFloat32).clone();
                        input = input.to(torch::TensorOptions().dtype(dtype).device(device));
                    }
                    break;
                case 1:
                    // All zeros (including negative zero)
                    input = torch::zeros(shape, torch::TensorOptions().dtype(dtype).device(device));
                    if (offset < size && data[offset++] % 2) {
                        input = -input;  // Create negative zeros
                    }
                    break;
                case 2:
                    // Mix of positive and negative values
                    input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
                    break;
                case 3:
                    // Special values (inf, -inf, nan)
                    {
                        input = torch::empty(shape, torch::TensorOptions().dtype(dtype).device(device));
                        if (dtype == torch::kFloat32 || dtype == torch::kFloat64 || 
                            dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
                            input.fill_(std::numeric_limits<float>::infinity());
                            if (offset < size && data[offset++] % 3 == 0) {
                                input = -input;
                            } else if (offset < size && data[offset++] % 3 == 1) {
                                input.fill_(std::numeric_limits<float>::quiet_NaN());
                            }
                        } else {
                            input.fill_(1);
                        }
                    }
                    break;
                case 4:
                    // Very small values near zero
                    input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device)) * 1e-30;
                    break;
                case 5:
                    // Mix including -0.0 and +0.0
                    {
                        input = torch::zeros(shape, torch::TensorOptions().dtype(dtype).device(device));
                        if (total_elements > 0) {
                            auto flat = input.flatten();
                            for (int64_t i = 0; i < flat.size(0); ++i) {
                                if (offset < size && data[offset++] % 2) {
                                    flat[i] = -0.0f;
                                }
                            }
                        }
                    }
                    break;
            }
        }
        
        // Test signbit operation
        torch::Tensor result;
        
        if (use_out_tensor && !shape.empty()) {
            // Test with out parameter
            torch::Tensor out = torch::empty(shape, torch::TensorOptions().dtype(torch::kBool).device(device));
            
            // Test with various out tensor configurations
            uint8_t out_config = (offset < size) ? data[offset++] : 0;
            switch (out_config % 3) {
                case 0:
                    // Correct shape
                    result = torch::signbit(input, out);
                    break;
                case 1:
                    // Try with wrong dtype out tensor (should convert or error)
                    try {
                        torch::Tensor wrong_out = torch::empty(shape, torch::TensorOptions().dtype(torch::kInt32).device(device));
                        result = torch::signbit(input, wrong_out);
                    } catch (...) {
                        // Expected for wrong dtype, fall back to regular call
                        result = torch::signbit(input);
                    }
                    break;
                case 2:
                    // Try with wrong shape out tensor
                    try {
                        std::vector<int64_t> wrong_shape = {1};
                        torch::Tensor wrong_out = torch::empty(wrong_shape, torch::TensorOptions().dtype(torch::kBool).device(device));
                        result = torch::signbit(input, wrong_out);
                    } catch (...) {
                        // Expected for wrong shape, fall back to regular call
                        result = torch::signbit(input);
                    }
                    break;
            }
        } else {
            // Test without out parameter
            result = torch::signbit(input);
        }
        
        // Verify result properties
        if (result.defined()) {
            // Check that result is boolean
            if (result.dtype() != torch::kBool) {
                std::cerr << "Warning: signbit result is not boolean type" << std::endl;
            }
            
            // Check shape matches input
            if (result.sizes() != input.sizes()) {
                std::cerr << "Warning: signbit result shape doesn't match input" << std::endl;
            }
            
            // Access some elements to ensure computation completed
            if (result.numel() > 0) {
                auto first = result.flatten()[0].item<bool>();
                (void)first;  // Suppress unused warning
            }
        }
        
        // Test edge cases with strided tensors
        if (offset < size && data[offset++] % 2 && input.dim() > 1) {
            torch::Tensor transposed = input.transpose(0, input.dim() - 1);
            torch::Tensor result2 = torch::signbit(transposed);
            (void)result2;
        }
        
        // Test with view
        if (offset < size && data[offset++] % 2 && input.numel() > 1) {
            torch::Tensor viewed = input.view({-1});
            torch::Tensor result3 = torch::signbit(viewed);
            (void)result3;
        }
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are often expected for invalid operations
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    catch (...)
    {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0; // keep the input
}