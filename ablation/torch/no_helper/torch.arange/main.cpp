#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 16) {
        return 0;  // Need minimum bytes for parameters
    }

    try {
        size_t offset = 0;
        
        // Extract start value (double)
        double start = 0.0;
        if (offset + sizeof(double) <= size) {
            std::memcpy(&start, data + offset, sizeof(double));
            offset += sizeof(double);
            // Clamp to reasonable range to avoid infinite loops
            if (std::isnan(start) || std::isinf(start)) {
                start = 0.0;
            } else if (start > 1e6) {
                start = 1e6;
            } else if (start < -1e6) {
                start = -1e6;
            }
        }

        // Extract end value (double)
        double end = 1.0;
        if (offset + sizeof(double) <= size) {
            std::memcpy(&end, data + offset, sizeof(double));
            offset += sizeof(double);
            // Clamp to reasonable range
            if (std::isnan(end) || std::isinf(end)) {
                end = start + 10.0;
            } else if (end > 1e6) {
                end = 1e6;
            } else if (end < -1e6) {
                end = -1e6;
            }
        }

        // Extract step value (double)
        double step = 1.0;
        if (offset + sizeof(double) <= size) {
            std::memcpy(&step, data + offset, sizeof(double));
            offset += sizeof(double);
            // Avoid zero/nan/inf step
            if (std::isnan(step) || std::isinf(step) || step == 0.0) {
                step = 1.0;
            } else if (std::abs(step) > 1e4) {
                step = (step > 0) ? 1e4 : -1e4;
            }
        }

        // Extract dtype choice
        uint8_t dtype_choice = 0;
        if (offset < size) {
            dtype_choice = data[offset++] % 8;
        }

        // Extract device choice
        uint8_t device_choice = 0;
        if (offset < size) {
            device_choice = data[offset++] % 2;
        }

        // Extract requires_grad flag
        bool requires_grad = false;
        if (offset < size) {
            requires_grad = (data[offset++] % 2) == 1;
        }

        // Set dtype based on fuzzer input
        torch::ScalarType dtype = torch::kFloat32;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kInt16; break;
            case 5: dtype = torch::kInt8; break;
            case 6: dtype = torch::kUInt8; break;
            case 7: dtype = torch::kFloat16; break;
            default: dtype = torch::kFloat32; break;
        }

        // Set device
        torch::Device device = torch::kCPU;
        if (device_choice == 1 && torch::cuda::is_available()) {
            device = torch::kCUDA;
        }

        // Limit the number of elements to prevent OOM
        double num_elements = std::abs((end - start) / step);
        if (num_elements > 100000) {
            // Adjust end to limit elements
            end = start + step * 100000;
        }

        // Test 1: Basic arange with all parameters
        {
            auto options = torch::TensorOptions()
                .dtype(dtype)
                .device(device)
                .requires_grad(requires_grad && (dtype == torch::kFloat32 || dtype == torch::kFloat64));
            
            torch::Tensor result = torch::arange(start, end, step, options);
            
            // Perform some operations to exercise the tensor
            if (result.numel() > 0) {
                auto sum = result.sum();
                if (result.numel() > 1) {
                    auto mean = result.mean();
                }
            }
        }

        // Test 2: arange with integer inputs for integer dtypes
        if (dtype == torch::kInt32 || dtype == torch::kInt64 || 
            dtype == torch::kInt16 || dtype == torch::kInt8) {
            int64_t int_start = static_cast<int64_t>(start);
            int64_t int_end = static_cast<int64_t>(end);
            int64_t int_step = static_cast<int64_t>(step);
            if (int_step == 0) int_step = 1;
            
            auto options = torch::TensorOptions()
                .dtype(dtype)
                .device(device);
            
            torch::Tensor result = torch::arange(int_start, int_end, int_step, options);
        }

        // Test 3: Single argument arange (end only)
        if (offset < size && end > 0) {
            auto options = torch::TensorOptions()
                .dtype(dtype)
                .device(device);
            
            torch::Tensor result = torch::arange(end, options);
        }

        // Test 4: Two argument arange (start, end)
        {
            auto options = torch::TensorOptions()
                .dtype(dtype)
                .device(device);
            
            torch::Tensor result = torch::arange(start, end, options);
        }

        // Test 5: Edge cases with negative step
        if (step < 0 && start > end) {
            auto options = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(device);
            
            torch::Tensor result = torch::arange(start, end, step, options);
        }

        // Test 6: Very small step values (floating point precision test)
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            double tiny_step = 1e-6;
            double tiny_end = start + tiny_step * 100;
            
            auto options = torch::TensorOptions()
                .dtype(dtype)
                .device(device);
            
            torch::Tensor result = torch::arange(start, tiny_end, tiny_step, options);
        }

    } catch (const c10::Error& e) {
        // PyTorch-specific exceptions are expected for invalid inputs
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}