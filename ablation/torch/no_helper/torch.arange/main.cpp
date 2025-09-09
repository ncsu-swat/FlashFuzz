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
        if (Size < 16) return 0;

        // Extract fuzzing parameters
        uint8_t variant = consume_uint8_t(Data, Size, offset);
        
        // Test different variants of torch.arange
        switch (variant % 8) {
            case 0: {
                // Test arange(end) - single parameter
                double end = consume_double(Data, Size, offset);
                if (offset >= Size) return 0;
                
                // Clamp end to reasonable range to avoid excessive memory usage
                end = std::clamp(end, -10000.0, 10000.0);
                
                auto result = torch::arange(end);
                break;
            }
            case 1: {
                // Test arange(start, end) - two parameters
                double start = consume_double(Data, Size, offset);
                double end = consume_double(Data, Size, offset);
                if (offset >= Size) return 0;
                
                // Clamp to reasonable ranges
                start = std::clamp(start, -10000.0, 10000.0);
                end = std::clamp(end, -10000.0, 10000.0);
                
                auto result = torch::arange(start, end);
                break;
            }
            case 2: {
                // Test arange(start, end, step) - three parameters
                double start = consume_double(Data, Size, offset);
                double end = consume_double(Data, Size, offset);
                double step = consume_double(Data, Size, offset);
                if (offset >= Size) return 0;
                
                // Clamp to reasonable ranges
                start = std::clamp(start, -10000.0, 10000.0);
                end = std::clamp(end, -10000.0, 10000.0);
                
                // Ensure step is not zero and not too small/large
                if (std::abs(step) < 1e-10 || std::abs(step) > 1000.0) {
                    step = (step >= 0) ? 1.0 : -1.0;
                }
                
                auto result = torch::arange(start, end, step);
                break;
            }
            case 3: {
                // Test with different dtypes
                double start = consume_double(Data, Size, offset);
                double end = consume_double(Data, Size, offset);
                uint8_t dtype_choice = consume_uint8_t(Data, Size, offset);
                if (offset >= Size) return 0;
                
                start = std::clamp(start, -1000.0, 1000.0);
                end = std::clamp(end, -1000.0, 1000.0);
                
                torch::ScalarType dtype;
                switch (dtype_choice % 6) {
                    case 0: dtype = torch::kFloat32; break;
                    case 1: dtype = torch::kFloat64; break;
                    case 2: dtype = torch::kInt32; break;
                    case 3: dtype = torch::kInt64; break;
                    case 4: dtype = torch::kInt16; break;
                    case 5: dtype = torch::kInt8; break;
                }
                
                auto options = torch::TensorOptions().dtype(dtype);
                auto result = torch::arange(start, end, options);
                break;
            }
            case 4: {
                // Test with device specification
                double end = consume_double(Data, Size, offset);
                uint8_t device_choice = consume_uint8_t(Data, Size, offset);
                if (offset >= Size) return 0;
                
                end = std::clamp(end, -1000.0, 1000.0);
                
                torch::Device device = (device_choice % 2 == 0) ? torch::kCPU : torch::kCPU; // Always CPU for safety
                auto options = torch::TensorOptions().device(device);
                auto result = torch::arange(end, options);
                break;
            }
            case 5: {
                // Test with requires_grad
                double start = consume_double(Data, Size, offset);
                double end = consume_double(Data, Size, offset);
                bool requires_grad = consume_uint8_t(Data, Size, offset) % 2;
                if (offset >= Size) return 0;
                
                start = std::clamp(start, -1000.0, 1000.0);
                end = std::clamp(end, -1000.0, 1000.0);
                
                auto options = torch::TensorOptions().requires_grad(requires_grad);
                auto result = torch::arange(start, end, options);
                break;
            }
            case 6: {
                // Test edge cases with integer parameters
                int32_t start = static_cast<int32_t>(consume_uint32_t(Data, Size, offset) % 2000 - 1000);
                int32_t end = static_cast<int32_t>(consume_uint32_t(Data, Size, offset) % 2000 - 1000);
                int32_t step = static_cast<int32_t>(consume_uint32_t(Data, Size, offset) % 20 + 1);
                if (offset >= Size) return 0;
                
                // Randomly make step negative
                if (consume_uint8_t(Data, Size, offset) % 2) step = -step;
                
                auto result = torch::arange(start, end, step);
                break;
            }
            case 7: {
                // Test with all options combined
                double start = consume_double(Data, Size, offset);
                double end = consume_double(Data, Size, offset);
                double step = consume_double(Data, Size, offset);
                uint8_t dtype_choice = consume_uint8_t(Data, Size, offset);
                bool requires_grad = consume_uint8_t(Data, Size, offset) % 2;
                if (offset >= Size) return 0;
                
                start = std::clamp(start, -500.0, 500.0);
                end = std::clamp(end, -500.0, 500.0);
                
                if (std::abs(step) < 1e-10 || std::abs(step) > 100.0) {
                    step = (step >= 0) ? 1.0 : -1.0;
                }
                
                torch::ScalarType dtype;
                switch (dtype_choice % 4) {
                    case 0: dtype = torch::kFloat32; break;
                    case 1: dtype = torch::kFloat64; break;
                    case 2: dtype = torch::kInt32; break;
                    case 3: dtype = torch::kInt64; break;
                }
                
                auto options = torch::TensorOptions()
                    .dtype(dtype)
                    .device(torch::kCPU)
                    .requires_grad(requires_grad);
                    
                auto result = torch::arange(start, end, step, options);
                break;
            }
        }

        // Additional edge case testing
        if (offset < Size) {
            uint8_t edge_case = consume_uint8_t(Data, Size, offset);
            switch (edge_case % 4) {
                case 0: {
                    // Test with very small step
                    auto result = torch::arange(0.0, 1.0, 0.1);
                    break;
                }
                case 1: {
                    // Test with negative range
                    auto result = torch::arange(10, 0, -1);
                    break;
                }
                case 2: {
                    // Test with zero range
                    auto result = torch::arange(5, 5);
                    break;
                }
                case 3: {
                    // Test with single element
                    auto result = torch::arange(1);
                    break;
                }
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}