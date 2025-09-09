#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least enough data for basic parameters
        if (Size < 20) return 0;

        // Extract start value (float)
        float start = extract_float(Data, Size, offset);
        
        // Extract end value (float)
        float end = extract_float(Data, Size, offset);
        
        // Extract steps (int64_t, must be positive)
        int64_t steps_raw = extract_int64(Data, Size, offset);
        int64_t steps = std::max(1LL, std::abs(steps_raw) % 10000 + 1); // Limit steps to reasonable range
        
        // Extract dtype choice
        uint8_t dtype_choice = extract_uint8(Data, Size, offset) % 8;
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kComplexFloat; break;
            case 5: dtype = torch::kComplexDouble; break;
            case 6: dtype = torch::kFloat16; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Extract device choice
        uint8_t device_choice = extract_uint8(Data, Size, offset) % 2;
        torch::Device device = (device_choice == 0) ? torch::kCPU : torch::kCUDA;
        
        // Extract requires_grad flag
        bool requires_grad = extract_uint8(Data, Size, offset) % 2 == 1;
        
        // Extract test variant
        uint8_t variant = extract_uint8(Data, Size, offset) % 6;
        
        torch::Tensor result;
        
        switch (variant) {
            case 0: {
                // Basic linspace with float values
                result = torch::linspace(start, end, steps);
                break;
            }
            case 1: {
                // Linspace with dtype specification
                result = torch::linspace(start, end, steps, torch::TensorOptions().dtype(dtype));
                break;
            }
            case 2: {
                // Linspace with device specification (skip CUDA if not available)
                if (device.is_cuda() && !torch::cuda::is_available()) {
                    device = torch::kCPU;
                }
                result = torch::linspace(start, end, steps, torch::TensorOptions().device(device));
                break;
            }
            case 3: {
                // Linspace with requires_grad
                result = torch::linspace(start, end, steps, torch::TensorOptions().requires_grad(requires_grad));
                break;
            }
            case 4: {
                // Linspace with tensor inputs for start and end
                torch::Tensor start_tensor = torch::tensor(start);
                torch::Tensor end_tensor = torch::tensor(end);
                result = torch::linspace(start_tensor, end_tensor, steps);
                break;
            }
            case 5: {
                // Linspace with all options
                if (device.is_cuda() && !torch::cuda::is_available()) {
                    device = torch::kCPU;
                }
                torch::TensorOptions options = torch::TensorOptions()
                    .dtype(dtype)
                    .device(device)
                    .requires_grad(requires_grad);
                result = torch::linspace(start, end, steps, options);
                break;
            }
        }
        
        // Test edge cases with remaining data
        if (offset < Size) {
            uint8_t edge_case = extract_uint8(Data, Size, offset) % 8;
            
            switch (edge_case) {
                case 0: {
                    // Test with steps = 1
                    torch::linspace(start, end, 1);
                    break;
                }
                case 1: {
                    // Test with start == end
                    torch::linspace(start, start, steps);
                    break;
                }
                case 2: {
                    // Test with very large steps (but reasonable)
                    int64_t large_steps = std::min(steps * 10, 50000LL);
                    torch::linspace(start, end, large_steps);
                    break;
                }
                case 3: {
                    // Test with negative to positive range
                    torch::linspace(-std::abs(start), std::abs(end), steps);
                    break;
                }
                case 4: {
                    // Test with very small range
                    float small_end = start + 1e-6f;
                    torch::linspace(start, small_end, steps);
                    break;
                }
                case 5: {
                    // Test with swapped start/end
                    torch::linspace(end, start, steps);
                    break;
                }
                case 6: {
                    // Test with zero values
                    torch::linspace(0.0f, end, steps);
                    break;
                }
                case 7: {
                    // Test with infinity handling (if finite)
                    if (std::isfinite(start) && std::isfinite(end)) {
                        torch::linspace(start, end, steps);
                    }
                    break;
                }
            }
        }
        
        // Verify result properties
        if (result.defined()) {
            // Check that result has correct size
            if (result.size(0) != steps) {
                std::cerr << "Unexpected tensor size" << std::endl;
            }
            
            // Check that result is 1D
            if (result.dim() != 1) {
                std::cerr << "Unexpected tensor dimensions" << std::endl;
            }
            
            // Access first and last elements to trigger any potential issues
            if (steps > 0) {
                auto first_val = result[0];
                if (steps > 1) {
                    auto last_val = result[steps - 1];
                }
            }
            
            // Test some tensor operations on result
            auto sum_val = result.sum();
            auto mean_val = result.mean();
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}